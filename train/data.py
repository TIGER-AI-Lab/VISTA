import torch
import yaml
import numpy as np
import bisect
import PIL
from PIL import Image
from decord import VideoReader
import os
import math
import random
import cv2

from pathlib import Path
from train.train_utils import load_json_data
from train.conversation import SeparatorStyle
from typing import List, Dict
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_ID = None # should be set when loading the processor
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_TOKEN_ID = None # should be set when loading the processor

def set_ignore_index(new_ignore_index=-100):
    global IGNORE_INDEX
    IGNORE_INDEX = new_ignore_index

def set_default_image_token(new_default_image_token="<image>"):
    global DEFAULT_IMAGE_TOKEN
    DEFAULT_IMAGE_TOKEN = new_default_image_token
    print("setting default image token to", new_default_image_token)

def set_default_image_token_id(new_default_image_token_id=None):
    global DEFAULT_IMAGE_TOKEN_ID
    DEFAULT_IMAGE_TOKEN_ID = new_default_image_token_id
    print("setting default image token id to", new_default_image_token_id)
    
def set_default_video_token(new_default_video_token="<video>"):
    global DEFAULT_VIDEO_TOKEN
    DEFAULT_VIDEO_TOKEN = new_default_video_token
    print("setting default video token to", new_default_video_token)
    
def set_default_video_token_id(new_default_video_token_id=None):
    global DEFAULT_VIDEO_TOKEN_ID
    DEFAULT_VIDEO_TOKEN_ID = new_default_video_token_id
    print("setting default video token id to", new_default_video_token_id)


def get_resize_output_image_size(height, width, shortest_edge, longest_edge):
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        The output size of the image after resizing.
    """

    min_len = shortest_edge
    max_len = longest_edge
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width

def load_image_from_path(image_path):
    if isinstance(image_path, Image.Image):
        image = image_path
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    return [image]

def get_frame_indices(num_frames, vlen, sample='rand', input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def tokenizer_image_token(prompt, tokenizer, image_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    image_token_index = tokenizer.convert_tokens_to_ids(image_token)
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(image_token)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


class CaptioningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processor,
        json_path,
        data_path,
        name,
        dataset_type,
        max_img_seq_len,
        max_txt_seq_len,
        conv_format,
        is_master_worker=True,
        do_resize=True,
        img_longest_edge=None,
        img_shortest_edge=None,
        max_size=None,
        shuffle=False,
        sample_ratio=1.0,
        # for video
        model_patch_size=16,
        num_tries=10,
        video_sample_type='rand',
        video_num_frames=8,
    ):
        from .data_utils import captioning_templates

        self.captioning_templates = captioning_templates
        self.processor = processor
        self.json_path = Path(json_path)
        self.dataset_type = dataset_type
        assert self.dataset_type in ['image', 'video'], f"Unknown dataset type {self.dataset_type}"
        self.name = name
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.do_resize = do_resize
        self.img_longest_edge = img_longest_edge
        self.img_shortest_edge = img_shortest_edge

        self.num_tries = num_tries
        self.video_sample_type = video_sample_type
        self.video_num_frames = video_num_frames

        self.print(f"Loading dataset '{name}' from {json_path}")
        self.data = load_json_data(json_path)
        self.data_dir = data_path
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        if sample_ratio < 1.0:
            self.print(f"Downsampling to {sample_ratio} of the data")
            self.max_size = int(len(self.data) * sample_ratio)
        if self.max_size:
            self.print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
            self.data = self.data[:self.max_size]

        self.conv = conv_format.copy()

        self.data_idx = [*range(len(self.data))]

        if sample_ratio > 1.0:
            additional_samples = int(len(self.data) * (sample_ratio - 1))
            self.print(f"Adding {additional_samples} samples for dataset {name}")
            added_idx = []
            while additional_samples > len(self.data_idx):
                added_idx.extend(self.data_idx)
                additional_samples -= len(self.data_idx)
            random.seed(42)
            added_idx.extend(random.sample(self.data_idx, additional_samples))
            self.data_idx += added_idx

        self.max_img_seq_len = max_img_seq_len
        self.max_txt_seq_len = max_txt_seq_len
        self.model_patch_size = model_patch_size
    
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)
        
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        real_idx = self.data_idx[idx]
        item = self.data[real_idx]
        data_path = item['image']
        data_path = os.path.join(self.data_dir, data_path)

        DEFAULT_TOKEN = None
        if self.dataset_type == 'image':
            sub_images = self.load_media_data_image(data_path)
            DEFAULT_TOKEN = DEFAULT_IMAGE_TOKEN
        elif self.dataset_type == 'video':
            sub_images, durations = self.load_media_data_video(data_path)
            DEFAULT_TOKEN = DEFAULT_VIDEO_TOKEN
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")

        conv = self.conv.copy()
        conv.messages = []
        user_template = random.choice(self.captioning_templates['user'])
        assistant_template = random.choice(self.captioning_templates['assistant'])
        conv.append_message(conv.roles[0], DEFAULT_TOKEN + ' ' + user_template.format(self.dataset_type))
        conv.append_message(conv.roles[1], assistant_template.format(self.dataset_type, item['caption']))
        conv_messages = conv.messages.copy()
        
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            # NOTE: this is for the pretraining, where we only use the pure text or interleaved text and images
            source = conv_messages
            assert len(source) == 2, "we only use the text in the second message for pretraining."
            # assert DEFAULT_IMAGE_TOKEN in source[0][1]
            # assert len(sub_images) == 1 if isinstance(sub_images, list) else isinstance(sub_images, PIL.Image.Image)
            if isinstance(sub_images, PIL.Image.Image):
                sub_images = [sub_images]
            text = source[1][1]
            image_token_count = source[1][1].count(DEFAULT_TOKEN)
            if image_token_count < len(sub_images):
                text = f"{DEFAULT_TOKEN} " * (len(sub_images) - image_token_count) + text
            conv_str = text + self.conv.sep
            encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, do_resize=self.do_resize, max_length=self.max_txt_seq_len)
        else:
            # NOTE: this is for the conversation style finetuning
            # check the number of images
            image_token_count = sum([message[1].count(DEFAULT_TOKEN) for message in conv_messages])
            if isinstance(sub_images, list):
                if image_token_count < len(sub_images):
                    conv_messages[0][1] = DEFAULT_TOKEN * (len(sub_images) - image_token_count) + conv_messages[0][1]
            else:
                if image_token_count < 1:
                    conv_messages[0][1] = DEFAULT_TOKEN + conv_messages[0][1]
            self.conv.messages = conv_messages
            conv_str = self.conv.get_prompt()
            encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, do_resize=self.do_resize, max_length=self.max_txt_seq_len)

        if "image_patches" in encoding:
            encoding.pop("attention_mask")
            encoding['image_patches'] = encoding['image_patches'][0] # todo
        
        if self.conv.sep_style == SeparatorStyle.KANGAROO:
            encoding['durations'] = torch.Tensor(durations).unsqueeze(0)
        
        if "labels" not in encoding:
            encoding["labels"] = torch.full_like(encoding["input_ids"], IGNORE_INDEX, dtype=encoding["input_ids"].dtype)
            input_ids = encoding["input_ids"][0]
            target = encoding["labels"][0]
            if self.conv.sep_style == SeparatorStyle.MFUYU:
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep2_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep2)
                
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist() 
                sep2_idxs = torch.nonzero((input_ids == sep2_id), as_tuple=True)[0].tolist() 
                if not (len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1):
                    torch.set_printoptions(profile="full")
                    raise ValueError(f"len({sep_idxs}) != len({sep2_idxs})")
                assert len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1, f"len({sep_idxs}) != len({sep2_idxs})"
                if len(sep_idxs) == len(sep2_idxs) + 1:
                    sep2_idxs.append(len(input_ids) - 1)
                for j in range(len(sep_idxs)):
                    target[sep_idxs[j]+1:sep2_idxs[j] + 1] = input_ids[sep_idxs[j]+1:sep2_idxs[j] + 1]
            elif self.conv.sep_style == SeparatorStyle.SINGLE or self.conv.sep_style == SeparatorStyle.LLAMA_3 or self.conv.sep_style == SeparatorStyle.FALCON or self.conv.sep_style == SeparatorStyle.QWEN2 or self.conv.sep_style == SeparatorStyle.KANGAROO or self.conv.sep_style == SeparatorStyle.PHI3_V:
                if self.conv.system != "":
                    skip_offset = 0
                else:
                    skip_offset = 1
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
                for i in range(len(sep_idxs)):
                    if i % 2 == skip_offset:
                        continue
                    if i == len(sep_idxs) - 1:
                        target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                    else:
                        target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
            elif self.conv.sep_style == SeparatorStyle.IDEFICS_2:
                if self.conv.system:
                    skip_offset = 0
                else:
                    skip_offset = 1
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
                for i in range(len(sep_idxs)):
                    if i % 2 == skip_offset:
                        continue
                    if i == len(sep_idxs) - 1:
                        target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                    else:
                        target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
            elif self.conv.sep_style == SeparatorStyle.PLAIN:
                assert DEFAULT_TOKEN is not None, "Please set the default image or video token id by calling set_default_image_token_id or set_default_video_token_id, this is required to masking the image or video tokens for pretraining."
                # mask the image tokens in the text
                target[input_ids != DEFAULT_TOKEN] = input_ids[input_ids != DEFAULT_TOKEN]
                # source = conv_str
                # tokenized_len = len(self.processor(source[0][1], sub_images, return_tensors="pt")["input_ids"][0])
                # target[tokenized_len:] = input_ids[tokenized_len:]
            elif self.conv.sep_style == SeparatorStyle.TWO:
                target = input_ids.clone()
                sep = self.conv.sep + self.conv.roles[1] + ": "

                total_len = int(target.ne(self.processor.tokenizer.pad_token_id).sum())

                rounds = conv_str.split(self.conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep

                    round_len = len(tokenizer_image_token(rou, self.processor.tokenizer, image_token=DEFAULT_TOKEN))
                    instruction_len = len(tokenizer_image_token(parts[0], self.processor.tokenizer, image_token=DEFAULT_TOKEN)) - 2

                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                target[cur_len:] = IGNORE_INDEX

                if cur_len < self.processor.tokenizer.model_max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_INDEX
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" (ignored)"
                        )
                encoding["labels"][0] = target
            else:
                raise ValueError(f"Unknown separator style {self.conv.sep_style}")
            # replace IGNORE_INDEX in target_ids with 0 and decode it, then print for debug
            if torch.all(target == IGNORE_INDEX):
                self.print("no labels for a sample in ", data_path, self.name)
            
        return encoding

    def load_media_data_image(self, data_path):
        for _ in range(self.num_tries):
            try:
                image = load_image_from_path(data_path)
        
                if not self.do_resize and (self.img_shortest_edge is not None and self.img_longest_edge is not None):
                    resized_image = []
                    for img in image:
                        height, width = get_resize_output_image_size(img.size[1], img.size[0], self.img_shortest_edge, self.img_longest_edge)
                        resized_image.append(img.resize((width, height), resample=3))
                    image = resized_image
                return image
            except Exception as e:
                self.print(
                    f"Caught exception {e} when loading image {data_path}, "
                    f"randomly sample a new image as replacement"
                )
                index = random.randint(0, len(self) - 1)
                item = self.data[index]
                data_path = item["image"]
                data_path = os.path.join(self.data_dir, data_path)
                continue
        else:
            raise RuntimeError(
                f"Failed to fetch image after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted images."
            )

    def load_media_data_video(self, data_path):
        for _ in range(self.num_tries):
            try:
                video_reader = VideoReader(data_path, num_threads=1)
                vlen = len(video_reader)
                fps = video_reader.get_avg_fps()
                if self.video_num_frames == 'auto':
                    if not self.do_resize:
                        vid = cv2.VideoCapture(data_path)
                        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height, width = get_resize_output_image_size(height, width, self.img_shortest_edge, self.img_longest_edge)
                        vid.release()
                    else:
                        height = self.processor.image_processor.size['height']
                        width = self.processor.image_processor.size['width']
                    num_patches = int((height // self.model_patch_size) * (width // self.model_patch_size))
                    video_num_frames = int(self.max_img_seq_len // num_patches)
                else:
                    width = height = None
                    video_num_frames = self.video_num_frames
                if self.do_resize:
                    # resize will be done in processor, skip resizing here
                    width = height = None
                video_num_frames = min(video_num_frames, vlen)
                frame_indices = get_frame_indices(video_num_frames, vlen, sample=self.video_sample_type, input_fps=fps)
                durations = [idx / fps  for idx in frame_indices]
                frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
                results = []
                for frame in frames:
                    img = PIL.Image.fromarray(frame, mode="RGB")
                    if width is not None and height is not None:
                        img = img.resize((int(width), int(height)), resample=3)
                    results.append(img)
            except Exception as e:
                self.print(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                index = random.randint(0, len(self) - 1)
                item = self.data[index]
                data_path = item["image"]
                data_path = os.path.join(self.data_dir, data_path)
                continue
            return [results], durations
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processor,
        json_path,
        data_path,
        name,
        dataset_type,
        max_img_seq_len,
        max_txt_seq_len,
        conv_format,
        is_master_worker=True,
        do_resize=True,
        img_longest_edge=None,
        img_shortest_edge=None,
        max_size=None,
        shuffle=False,
        sample_ratio=1.0,
        # for video
        model_patch_size=16,
        num_tries=10,
        video_sample_type='rand',
        video_num_frames=8,
    ):
        from .data_utils import captioning_templates

        self.captioning_templates = captioning_templates
        self.processor = processor
        self.json_path = Path(json_path)
        self.dataset_type = dataset_type
        assert self.dataset_type in ['image', 'video'], f"Unknown dataset type {self.dataset_type}"
        self.name = name
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.do_resize = do_resize
        self.img_longest_edge = img_longest_edge
        self.img_shortest_edge = img_shortest_edge

        self.num_tries = num_tries
        self.video_sample_type = video_sample_type
        self.video_num_frames = video_num_frames

        self.print(f"Loading dataset '{name}' from {json_path}")
        self.data = load_json_data(json_path)
        self.data_dir = data_path
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        if sample_ratio < 1.0:
            self.print(f"Downsampling to {sample_ratio} of the data")
            self.max_size = int(len(self.data) * sample_ratio)
        if self.max_size:
            self.print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
            self.data = self.data[:self.max_size]

        self.conv = conv_format.copy()

        self.data_idx = [*range(len(self.data))]

        if sample_ratio > 1.0:
            additional_samples = int(len(self.data) * (sample_ratio - 1))
            self.print(f"Adding {additional_samples} samples for dataset {name}")
            added_idx = []
            while additional_samples > len(self.data_idx):
                added_idx.extend(self.data_idx)
                additional_samples -= len(self.data_idx)
            random.seed(42)
            added_idx.extend(random.sample(self.data_idx, additional_samples))
            self.data_idx += added_idx

        self.max_img_seq_len = max_img_seq_len
        self.max_txt_seq_len = max_txt_seq_len
        self.model_patch_size = model_patch_size
    
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)
        
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        real_idx = self.data_idx[idx]
        item = self.data[real_idx]
        data_path = item['image']
        conversations = item['conversations']
        data_path = os.path.join(self.data_dir, data_path)
        # print(self.name, data_path)
        DEFAULT_TOKEN = None
        if self.dataset_type == 'image':
            sub_images = self.load_media_data_image(data_path)
            for i, image in enumerate(sub_images):
                if image.size[0] < 16 or image.size[1] < 16:
                    scale_factor = max(16 / image.size[0], 16 / image.size[1])
                    sub_images[i] = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))).convert("RGB")
            DEFAULT_TOKEN = DEFAULT_IMAGE_TOKEN
        elif self.dataset_type == 'video':
            sub_images, durations = self.load_media_data_video(data_path)

            for i, image in enumerate(sub_images):
                for j, frame in enumerate(image):
                    if frame.size[0] < 16 or frame.size[1] < 16:
                        scale_factor = max(16 / frame.size[0], 16 / frame.size[1])
                        sub_images[i][j] = frame.resize((int(frame.size[0] * scale_factor), int(frame.size[1] * scale_factor))).convert("RGB")
            DEFAULT_TOKEN = DEFAULT_VIDEO_TOKEN
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")

        conv = self.conv.copy()
        conv.messages = []
        for i, sentence in enumerate(conversations):
            if sentence['from'] == 'human':
                if i == 0:
                    conv.append_message(conv.roles[0], DEFAULT_TOKEN + ' ' + sentence['value'])
                else:
                    conv.append_message(conv.roles[0], sentence['value'])
            elif sentence['from'] == 'gpt':
                conv.append_message(conv.roles[1], sentence['value'])
        conv_messages = conv.messages.copy()
        
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            # NOTE: this is for the pretraining, where we only use the pure text or interleaved text and images
            source = conv_messages
            assert len(source) == 2, "we only use the text in the second message for pretraining."
            # assert DEFAULT_IMAGE_TOKEN in source[0][1]
            # assert len(sub_images) == 1 if isinstance(sub_images, list) else isinstance(sub_images, PIL.Image.Image)
            if isinstance(sub_images, PIL.Image.Image):
                sub_images = [sub_images]
            text = source[1][1]
            image_token_count = source[1][1].count(DEFAULT_TOKEN)
            if image_token_count < len(sub_images):
                text = f"{DEFAULT_TOKEN} " * (len(sub_images) - image_token_count) + text
            conv_str = text + self.conv.sep
            encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, do_resize=self.do_resize, max_length=self.max_txt_seq_len)
        else:
            # NOTE: this is for the conversation style finetuning
            # check the number of images
            image_token_count = sum([message[1].count(DEFAULT_TOKEN) for message in conv_messages])
            if isinstance(sub_images, list):
                if image_token_count < len(sub_images):
                    conv_messages[0][1] = DEFAULT_TOKEN * (len(sub_images) - image_token_count) + conv_messages[0][1]
            else:
                if image_token_count < 1:
                    conv_messages[0][1] = DEFAULT_TOKEN + conv_messages[0][1]
            self.conv.messages = conv_messages
            conv_str = self.conv.get_prompt()
            encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, do_resize=self.do_resize, max_length=self.max_txt_seq_len)

        if "image_patches" in encoding:
            encoding.pop("attention_mask")
            encoding['image_patches'] = encoding['image_patches'][0] # todo
        
        if self.conv.sep_style == SeparatorStyle.KANGAROO:
            encoding['durations'] = torch.Tensor(durations).unsqueeze(0)
        
        if "labels" not in encoding:
            encoding["labels"] = torch.full_like(encoding["input_ids"], IGNORE_INDEX, dtype=encoding["input_ids"].dtype)
            input_ids = encoding["input_ids"][0]
            target = encoding["labels"][0]
            if self.conv.sep_style == SeparatorStyle.MFUYU:
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep2_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep2)
                
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist() 
                sep2_idxs = torch.nonzero((input_ids == sep2_id), as_tuple=True)[0].tolist() 
                if not (len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1):
                    torch.set_printoptions(profile="full")
                    raise ValueError(f"len({sep_idxs}) != len({sep2_idxs})")
                assert len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1, f"len({sep_idxs}) != len({sep2_idxs})"
                if len(sep_idxs) == len(sep2_idxs) + 1:
                    sep2_idxs.append(len(input_ids) - 1)
                for j in range(len(sep_idxs)):
                    target[sep_idxs[j]+1:sep2_idxs[j] + 1] = input_ids[sep_idxs[j]+1:sep2_idxs[j] + 1]
            elif self.conv.sep_style == SeparatorStyle.SINGLE or self.conv.sep_style == SeparatorStyle.LLAMA_3 or self.conv.sep_style == SeparatorStyle.FALCON or self.conv.sep_style == SeparatorStyle.QWEN2 or self.conv.sep_style == SeparatorStyle.KANGAROO or self.conv.sep_style == SeparatorStyle.PHI3_V:
                if self.conv.system != "":
                    skip_offset = 0
                else:
                    skip_offset = 1
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
                for i in range(len(sep_idxs)):
                    if i % 2 == skip_offset:
                        continue
                    if i == len(sep_idxs) - 1:
                        target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                    else:
                        target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
            elif self.conv.sep_style == SeparatorStyle.IDEFICS_2:
                if self.conv.system:
                    skip_offset = 0
                else:
                    skip_offset = 1
                sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
                sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
                for i in range(len(sep_idxs)):
                    if i % 2 == skip_offset:
                        continue
                    if i == len(sep_idxs) - 1:
                        target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                    else:
                        target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
            elif self.conv.sep_style == SeparatorStyle.PLAIN:
                assert DEFAULT_TOKEN is not None, "Please set the default image or video token id by calling set_default_image_token_id or set_default_video_token_id, this is required to masking the image or video tokens for pretraining."
                # mask the image tokens in the text
                target[input_ids != DEFAULT_TOKEN] = input_ids[input_ids != DEFAULT_TOKEN]
                # source = conv_str
                # tokenized_len = len(self.processor(source[0][1], sub_images, return_tensors="pt")["input_ids"][0])
                # target[tokenized_len:] = input_ids[tokenized_len:]
            elif self.conv.sep_style == SeparatorStyle.TWO:
                target = input_ids.clone()
                sep = self.conv.sep + self.conv.roles[1] + ": "

                total_len = int(target.ne(self.processor.tokenizer.pad_token_id).sum())

                rounds = conv_str.split(self.conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break
                    has_image = DEFAULT_TOKEN in rou
                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep

                    if has_image:
                        round_len = len(tokenizer_image_token(rou, self.processor.tokenizer, image_token=DEFAULT_TOKEN))
                        instruction_len = len(tokenizer_image_token(parts[0], self.processor.tokenizer, image_token=DEFAULT_TOKEN)) - 2
                    else:
                        round_len = len(self.processor.tokenizer(rou).input_ids) - 1
                        instruction_len = len(self.processor.tokenizer(parts[0]).input_ids) - 3

                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                target[cur_len:] = IGNORE_INDEX

                if cur_len < self.processor.tokenizer.model_max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_INDEX
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" (ignored)"
                        )
                encoding["labels"][0] = target
            else:
                raise ValueError(f"Unknown separator style {self.conv.sep_style}")
            # replace IGNORE_INDEX in target_ids with 0 and decode it, then print for debug
            if torch.all(target == IGNORE_INDEX):
                self.print("no labels for a sample in ", data_path, self.name)
        return encoding

    def load_media_data_image(self, data_path):
        for _ in range(self.num_tries):
            try:
                image = load_image_from_path(data_path)
        
                if not self.do_resize and (self.img_shortest_edge is not None and self.img_longest_edge is not None):
                    resized_image = []
                    for img in image:
                        height, width = get_resize_output_image_size(img.size[1], img.size[0], self.img_shortest_edge, self.img_longest_edge)
                        resized_image.append(img.resize((width, height), resample=3))
                    image = resized_image
                return image
            except Exception as e:
                self.print(
                    f"Caught exception {e} when loading image {data_path}, "
                    f"randomly sample a new image as replacement"
                )
                index = random.randint(0, len(self) - 1)
                item = self.data[index]
                data_path = item["image"]
                data_path = os.path.join(self.data_dir, data_path)
                continue
        else:
            raise RuntimeError(
                f"Failed to fetch image after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted images."
            )

    def load_media_data_video(self, data_path):
        for _ in range(self.num_tries):
            try:
                video_reader = VideoReader(data_path, num_threads=1)
                vlen = len(video_reader)
                fps = video_reader.get_avg_fps()
                if self.video_num_frames == 'auto':
                    if not self.do_resize:
                        vid = cv2.VideoCapture(data_path)
                        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height, width = get_resize_output_image_size(height, width, self.img_shortest_edge, self.img_longest_edge)
                        vid.release()
                    else:
                        height = self.processor.image_processor.size['height']
                        width = self.processor.image_processor.size['width']
                    num_patches = int((height // self.model_patch_size) * (width // self.model_patch_size))
                    video_num_frames = int(self.max_img_seq_len // num_patches)
                else:
                    height = width = None
                    video_num_frames = self.video_num_frames
                if self.do_resize:
                    # resize will be done in processor, skip resizing here
                    width = height = None
                video_num_frames = min(video_num_frames, vlen)
                frame_indices = get_frame_indices(video_num_frames, vlen, sample=self.video_sample_type, input_fps=fps)
                durations = [idx / fps  for idx in frame_indices]
                frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
                results = []
                for frame in frames:
                    img = PIL.Image.fromarray(frame, mode="RGB")
                    if width is not None and height is not None:
                        img = img.resize((int(width), int(height)), resample=3)
                    results.append(img)
            except Exception as e:
                self.print(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                index = random.randint(0, len(self) - 1)
                item = self.data[index]
                data_path = item["image"]
                data_path = os.path.join(self.data_dir, data_path)
                continue
            return [results], durations
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )


class DatasetCollection(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset], balancing=False):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_len = sum(self.lengths)
        print("Total length of the dataset collection:", self.total_len)
        if balancing:
            sqrt_lengths = [math.sqrt(length) for length in self.lengths]
            sum_sqrt_lengths = sum(sqrt_lengths)
            sampling_probs = [sqrt_length / sum_sqrt_lengths for sqrt_length in sqrt_lengths]
            self._lengths = [int(self.total_len * min(prob * 1.1, 1)) for prob in sampling_probs]
            self.total_len = sum(self._lengths)
            self.cum_lengths = [0] + list(np.cumsum(self._lengths))
        else:
            self.cum_lengths = [0] + list(np.cumsum(self.lengths))
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        sub_idx = (idx - self.cum_lengths[dataset_idx]) % self.lengths[dataset_idx]
        return self.datasets[dataset_idx][sub_idx]
    

class Collator():
    def __init__(self, max_length=None, extra_collator_func=None):
        self.max_length = max_length
        self.extra_collator_func = extra_collator_func
    
    @staticmethod
    def _right_pad_inputs_with_attention_mask(model_inputs: List[Dict]):
        results = {}
        assert len(model_inputs) == 1, "This method only supports a single input, but get {} inputs".format(len(model_inputs))
        for k in model_inputs[0].keys():
            if k == "pixel_values" and isinstance(model_inputs[0][k], list):
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            elif k == "images" and isinstance(model_inputs[0][k], list):
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            elif k == "modalities" or k == "image_sizes":
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            elif model_inputs[0][k] is not None:
                results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
            else:
                results[k] = None
        return results
    
    @staticmethod
    def _right_pad_inputs_with_attention_mask_phi3_v(model_inputs: List[Dict]):
        results = {}
        assert len(model_inputs) == 1, "This method only supports a single input, but get {} inputs".format(len(model_inputs))
        for k in model_inputs[0].keys():
            if k == "pixel_values" and isinstance(model_inputs[0][k], list):
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            elif model_inputs[0][k] is not None:
                results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
            else:
                results[k] = None
        return results
    
    def __call__(self, batch):
        if self.extra_collator_func is not None:
            batch = self.extra_collator_func(batch)
        
        return batch
    
    @staticmethod
    def data_collator(examples, padding_value=0, max_length=2048):
        def trim_and_pad(seq, batch_first, padding_value):
            from torch.nn.utils.rnn import pad_sequence
            return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

        input_ids = trim_and_pad(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=padding_value,
        )
        position_ids = trim_and_pad(
            [example["position_ids"] for example in examples],
            batch_first=True,
            padding_value=padding_value,
        )
        targets = trim_and_pad(
            [example["labels"] for example in examples],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = trim_and_pad(
            [example["attention_mask"] for example in examples],
            batch_first=True,
            padding_value=padding_value,
        )
        pixel_values = [example["pixel_values"] for example in examples]
        image_bound = [example["image_bound"] for example in examples]
        tgt_sizes = [example["tgt_sizes"] for example in examples]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": targets,
            "attention_mask": attention_mask,
            "image_bound": image_bound,
            "tgt_sizes": tgt_sizes,
            "pixel_values": pixel_values,
        }

def load_data_from_config(data_args, processor):
    """
    Returns:
        all_datasets: Dict[str, List[Dataset]], mapping from split to list of datasets
        collator_fn: Callable
    """
    with open(data_args.data_config_file, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    print("Max Image/Video Context Length:", data_args.max_img_seq_len)
    print("Max Text Context Length:", data_args.max_txt_seq_len)
    all_datasets = {}
    for sub_dataset_config in data_config['data']:
        json_path = sub_dataset_config['json_path']
        data_path = sub_dataset_config['data_path']
        name = sub_dataset_config['name']
        dataset_type = sub_dataset_config['dataset_type']
        max_img_seq_len = sub_dataset_config.get('max_img_seq_len', data_args.max_img_seq_len)
        max_txt_seq_len = sub_dataset_config.get('max_txt_seq_len', data_args.max_txt_seq_len)
        max_size = sub_dataset_config.get('max_size', None)
        shuffle = sub_dataset_config.get('shuffle', False)
        do_resize = sub_dataset_config.get('img_do_resize', False)
        sample_ratio = sub_dataset_config.get('sample_ratio', 1.0)
        num_tries = sub_dataset_config.get('num_tries', 5)
        video_sample_type = sub_dataset_config.get('video_sample_type', 'rand')
        video_num_frames = sub_dataset_config.get('video_num_frames', 'auto')
        split = sub_dataset_config.get('split', 'train')
        img_longest_edge = sub_dataset_config.get('img_longest_edge', None)
        img_shortest_edge = sub_dataset_config.get('img_shortest_edge', None)
        if sub_dataset_config['format'] == 'caption':
            sub_dataset = CaptioningDataset(
                processor=processor,
                json_path=json_path,
                data_path=data_path,
                name=name,
                dataset_type=dataset_type,
                max_img_seq_len=max_img_seq_len,
                max_txt_seq_len=max_txt_seq_len,
                conv_format=data_args.conv_format,
                is_master_worker=data_args.is_master_worker,
                do_resize=do_resize,
                img_longest_edge=img_longest_edge,
                img_shortest_edge=img_shortest_edge,
                max_size=max_size,
                shuffle=shuffle,
                sample_ratio=sample_ratio,
                # for video
                model_patch_size=data_args.model_patch_size,
                num_tries=num_tries,
                video_sample_type=video_sample_type,
                video_num_frames=video_num_frames,
            )
        elif sub_dataset_config['format'] == 'conversation':
            sub_dataset = ConversationDataset(
                processor=processor,
                json_path=json_path,
                data_path=data_path,
                name=name,
                dataset_type=dataset_type,
                max_img_seq_len=max_img_seq_len,
                max_txt_seq_len=max_txt_seq_len,
                conv_format=data_args.conv_format,
                is_master_worker=data_args.is_master_worker,
                do_resize=do_resize,
                img_longest_edge=img_longest_edge,
                img_shortest_edge=img_shortest_edge,
                max_size=max_size,
                shuffle=shuffle,
                sample_ratio=sample_ratio,
                # for video
                model_patch_size=data_args.model_patch_size,
                num_tries=num_tries,
                video_sample_type=video_sample_type,
                video_num_frames=video_num_frames,
            )
        else:
            raise ValueError(f"Unknown data format {sub_dataset_config['format']}")
        if split not in all_datasets:
            all_datasets[split] = []
        all_datasets[split].append(sub_dataset)
    collator_fn = Collator(max_length=data_args.max_img_seq_len, extra_collator_func=data_args.extra_collator_func)
    
    if 'train' in all_datasets:
        if len(all_datasets['train']) == 1:
            train_dataset = all_datasets['train'][0]
        else:
            train_dataset = DatasetCollection(all_datasets['train'], data_args.dataset_balancing) if 'train' in all_datasets else None
    val_dataset = DatasetCollection(all_datasets['val'], data_args.dataset_balancing) if 'val' in all_datasets else None
    test_dataset = DatasetCollection(all_datasets['test'], data_args.dataset_balancing) if 'test' in all_datasets else None
    return train_dataset, val_dataset, test_dataset, collator_fn
