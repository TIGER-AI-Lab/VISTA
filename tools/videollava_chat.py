"""pip install transformers>=4.35.2
"""
import os
import torch
from typing import List, Optional
from transformers.utils import is_flash_attn_2_available
from models.videollava import VideoLlavaForConditionalGeneration, AdaptedVideoLlavaProcessor

from train.conversation import conv_templates
from tools.chat_utils import load_media_data_image, load_media_data_video, load_identity


def load_image_or_video(image_or_video, model, processor):
    _type = image_or_video["type"]
    content = image_or_video["content"]
    metadata = image_or_video.get("metadata", {})

    if _type == "image":
        load_func = load_media_data_image
    elif _type == "video":
        load_func = load_media_data_video
    elif _type == "pil_image" or _type == "pil_video":
        load_func = load_identity
    else:
        raise ValueError(f"Unknown type: {_type}")
    return load_func(content, model, processor, **metadata)

class VideoLLaVA():
    def __init__(self, model_path="LanguageBind/Video-LLaVA-7B-hf", device="cuda") -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, _attn_implementation=attn_implementation).to(device).eval()
        self.processor = AdaptedVideoLlavaProcessor.from_pretrained(model_path)
        self.patch_size = self.model.config.vision_config.patch_size
        self.conv = conv_templates["videollava"]
        self.terminators = [
            self.processor.tokenizer.eos_token_id,
        ]
        
    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] == "image" or x["type"] == "video" or x["type"] == "pil_image" or x["type"] == "pil_video"]
        assert len(images) == 1, "only support 1 input image/video"
        images = images[0]

        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        conv = self.conv.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        input_images = load_image_or_video(images, self.patch_size, self.processor)

        do_resize = False if images.get("metadata", True) else images["metadata"].get("do_resize", False)
        inputs = self.processor(text=prompt, input_images=input_images, do_resize=do_resize, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_config = generation_config if generation_config is not None else {}
        if "max_new_tokens" not in generation_config:
            generation_config["max_new_tokens"] = 512
        if "eos_token_id" not in generation_config:
            generation_config["eos_token_id"] = self.terminators
        
        generate_ids = self.model.generate(**inputs, **generation_config)
        generated_text = self.processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return generated_text

    
if __name__ == "__main__":
    model = VideoLLaVA(model_path="/cpfs/data/user/weiming/checkpoints/LanguageBind/Video-LLaVA-7B-hf")
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_1.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "img_longest_edge": 384,
                "max_img_seq_len": 16000,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> what animal is eating grass in the video? Answer with the option's letter from the given choices directly. Choose one option from: A. cat, B. panda, C. kangaroo, D. elephant."
        }
    ]
    print(model(test_input))
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_1.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "max_img_seq_len": 16500,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> what is shown in the video?"
        }
    ]
    print(model(test_input))
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_1.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "max_img_seq_len": 16500,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> describe the video in detail."
        }
    ]
    print(model(test_input))
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_2.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "max_img_seq_len": 16500,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> describe the video in detail."
        }
    ]
    print(model(test_input))
    print("############################################################################################")
    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_3.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "max_img_seq_len": 16500,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> describe the video in detail."
        }
    ]
    print(model(test_input))
    print("############################################################################################")

    test_input = [
        {
            "type": "video",
            "content": "/cpfs/data/user/weiming/repos/video-mamba/videomamba/videollm/output/assets/test_video_4.mp4",
            "metadata": {
                "video_num_frames": 8,
                "video_sample_type": "rand",
                "max_img_seq_len": 16500,
                "do_resize": False,
            }
        },
        {
            "type": "text",
            "content": "<video> describe the video in detail."
        }
    ]
    print(model(test_input))
    print("############################################################################################")