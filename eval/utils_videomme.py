import torch
import cv2
import copy
import os.path as osp
import pysubs2

from PIL import Image
from pandas import read_parquet
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from train.data import get_resize_output_image_size

def uniform_indices(num_frames: int, total_frames: int) -> list[int]:
    """Get uniform indices 

    Args:
        num_frames (int): number of frames
        total_frames (int): total number of frames

    Returns:
        list[int]: Output frame indices
    """
    if num_frames < total_frames:
        splits = torch.linspace(0, total_frames, num_frames+1, dtype=int)
        indices = ((splits[:-1] + splits[1:]) // 2).tolist()
    else:
        indices = list(range(total_frames))

    return indices


def fps_indices(input_fps: float, total_frames: int, output_fps: float = None, max_num_frames: int = -1) -> list[int]:
    """Get indices according to the output_fps

    Args:
        input_fps (float): input fps
        total_frames (int): total number of frames
        output_fps (float, optional): output fps. Defaults to None, means output_fps==input_fps.
        max_num_frames (int, optional): max number of frames. Defaults to -1, means no limitation.

    Returns:
        list[int]: Output frame indices
    """
    delta = 1 if output_fps is None else input_fps / output_fps
    indices = torch.arange(0, total_frames, delta).round().to(int)
    indices = [e for e in indices if e < total_frames]
    if 0 < max_num_frames < len(indices):
        indices = indices[:max_num_frames]

    return indices

def load_subtitle(sub_path: str, indices: list[int], fps: float) -> str:
    """Load subtitle related to given indices

    Args:
        sub_path (str): subtitle path
        indices (list[int]): frame indices
        fps (float): video average fps

    Returns:
        str: subtitle
    """
    subs = pysubs2.load(sub_path, encoding='utf-8')
    subtitles = []
    for idx in indices:
        sub_text = []
        cur_time = pysubs2.make_time(fps=fps, frames=idx)
        for sub in subs:
            if sub.end < cur_time:
                continue
            elif sub.start < cur_time:
                sub_text.append(sub.text.replace('\\N', ' '))
                break   # in accordance with the official Video-MME Benchmark
            else:
                break
        sub_text = ' '.join(sub_text)
        if sub_text.strip():
            subtitles.append(sub_text)
    subtitles = '\n'.join(subtitles)

    return subtitles
    

def load_decord(src_path: str, sample_type: str, sub_path: str = None, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    vr = VideoReader(src_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    do_resize = kwargs.pop('do_resize', False)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        width = height = None
        if num_frames == "auto":
            model_patch_size = kwargs['model_patch_size']
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            max_img_seq_len = kwargs['max_img_seq_len']
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
            num_patches = int((height // model_patch_size) * (width // model_patch_size))
            num_frames = int(max_img_seq_len // num_patches)
        if do_resize:
            vid = cv2.VideoCapture(src_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
            img_shortest_edge = kwargs['img_shortest_edge']
            img_longest_edge = kwargs['img_longest_edge']
            height, width = get_resize_output_image_size(height, width, img_shortest_edge, img_longest_edge)
        input_fps = float(vr.get_avg_fps())
        indices = uniform_indices(num_frames, total_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame).resize((int(width), int(height)), resample=3) if width and height else Image.fromarray(frame) for frame in frames]
    elif sample_type == 'fps':
        input_fps = float(vr.get_avg_fps())
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(input_fps, total_frames, output_fps, max_num_frames)
        durations = [idx / input_fps for idx in indices]
        frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
        frames = [Image.fromarray(frame) for frame in frames]
    else:
        raise ValueError(f'Do not support {sample_type} sample type')

    if sub_path is None:
        return frames, durations
    elif osp.exists(sub_path):
        subtitles = load_subtitle(sub_path, indices=indices, fps=float(vr.get_avg_fps()))
        return frames, durations, subtitles
    else:
        return frames, durations, ''


class VideoMMEDataset(Dataset):
    """Video-MME dataset. By default, videos are saved in `data/` and subtitles are saved in `subtitle/`
    """

    def __init__(self, dataset_path: str, sample_config: dict, use_subtitle: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_config = sample_config
        self.use_subtitle = use_subtitle

        data_dict = {}
        index_keys = ['video_id', 'duration', 'domain', 'sub_category', 'videoID']
        value_keys = ['question_id', 'task_type', 'question', 'options', 'answer']
        df = read_parquet(osp.join(dataset_path, 'videomme', 'test-00000-of-00001.parquet'))
        df['options'] = df['options'].apply(list)
        for _, data in df.iterrows():
            key = tuple(data[k] for k in index_keys)
            value = data[value_keys].to_dict()
            if key in data_dict:
                data_dict[key].append(value)
            else:
                data_dict[key] = [value]
        self.data_list = [dict(zip(index_keys + ['questions'], list(k) + [v])) for k, v in data_dict.items()]
        self.full_data_list = []
        for data in self.data_list:
            for question in data['questions']:
                d = copy.deepcopy(data)
                d['questions'] = [question]
                self.full_data_list.append(d)

    def __len__(self):
        return len(self.full_data_list)

    def __getitem__(self, idx) -> dict:
        if self.use_subtitle:
            frames, durations, subtitles = load_decord(
                src_path=osp.join(self.dataset_path, 'data', self.full_data_list[idx]['videoID'] + '.mp4'),
                sub_path=osp.join(self.dataset_path, 'subtitle', self.full_data_list[idx]['videoID'] + '.srt'),
                **self.sample_config
            )
            text = ['\n'.join([
                "This video's subtitles are listed below:",
                subtitles,
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.full_data_list[idx]['questions']]
        else:
            frames, durations = load_decord(
                src_path=osp.join(self.dataset_path, 'data', self.full_data_list[idx]['videoID'] + '.mp4'),
                **self.sample_config
            )
            text = ['\n'.join([
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.full_data_list[idx]['questions']]
            subtitles = ''

        return dict(
            video=frames,
            durations=torch.Tensor(durations),
            text=text,
            meta=self.full_data_list[idx]
        )