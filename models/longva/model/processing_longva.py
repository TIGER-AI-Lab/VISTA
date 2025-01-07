"""
Processor class for LongVA.
"""

import os
import json
from typing import List, Optional, Union, Dict

from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.processing_utils import transformers_module
from transformers.utils import IMAGE_PROCESSOR_NAME

from PIL import Image
import logging
import torch
import numpy as np

from models.longva.constants import IMAGE_TOKEN_INDEX
from models.longva.mm_utils import tokenizer_image_token, process_images

logger = logging.getLogger(__name__)


class LongVAProcessor:
    def __init__(self, image_processor=None, tokenizer=None, model_config=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        
    def preprocess_interleaved_images_and_text(
        self,
        text,
        images=None,
    ):
        assert text is not None, "text cannot be None."
        
        mode = None
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            if isinstance(images, list) and isinstance(images[0], Image.Image):
                mode = "<image>"
            elif isinstance(images, list) and isinstance(images[0], list):
                mode = "<video>"
            else:
                raise ValueError("Invalid input images. images must be a PIL image or a list of PIL images (images), or a list of list of PIL images (videos).")
            
            if isinstance(text, str):
                text = [text]
            elif isinstance(text, list):
                if len(text) != len(images):
                    raise ValueError("Invalid input text. Number of texts does not match number of images/videos.")
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")

            for i, t in enumerate(text):
                num_image_tokens = t.count(mode)
                if num_image_tokens < 1:
                    # prepend empty image tokens to text
                    if "<|start_header_id|>user<|end_header_id|>\n\n" in t:
                        t = t.replace("<|start_header_id|>user<|end_header_id|>\n\n", f"<|start_header_id|>user<|end_header_id|>\n\n{mode}\n", 1)
                    else:
                        t = mode + '\n' + t
                elif num_image_tokens > 1:
                    t = t.split(mode)
                    for j, s in enumerate(t):
                        if j < 1:
                            t[j] = s + mode
                    t = "".join(t)
                    logger.warning(f"Number of {mode} tokens: {num_image_tokens} exceeds 1. Automatically removing extra tokens at the end of the text.")
                if f"{mode} " in t:
                    t = t.replace(f"{mode} ", f"{mode}\n", 1)
                if mode in t:
                    replace_token = ""
                    num_segments = len(images[i])
                    ns = num_segments
                    ns = ns // 2 - 1
                    for _ in range(ns):
                        replace_token += "<image>"
                        replace_token += "<eof>"
                    replace_token += "<image>"
                    replace_token += "<eov>"
                    
                    replace_token = '<vi_start>' + replace_token + '<vi_end>'
                    t = t.replace(mode, replace_token)
                text[i] = t
            texts = text
            
        else:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
        
        return texts, images, mode

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        do_resize=True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        add_image_ids: bool = True,
    ) -> BatchFeature:
        assert isinstance(text, str), "text must be a string"
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            mode = "image"
        elif isinstance(images, list) and isinstance(images[0], list):
            mode = "video"
        else:
            raise ValueError("Invalid input images. images must be a PIL image or a list of PIL images (images), or a list of list of PIL images (videos).")

        if "<image> " in text:
            text = text.replace("<image> ", "<image>\n", 1)
        input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        # text_inputs: 
        # 1. input_ids: [batch_size, sequence_length], e.g. [1, 6]
        # 2. attention_mask: [batch_size, sequence_length], e.g. [1, 6]
        
        # check the number of image token ids, and truncated the number of images if needed
        pixel_values = None
        image_sizes = None
        if images is not None:
            if mode == "image":
                pixel_values = process_images(images, self.image_processor, self.model_config)
                image_sizes = images[0].size
            elif mode == "video": # video
                # flatten images
                videos = []
                for video in images:
                    np_video = np.array([np.array(img) for img in video])
                    pixel_values = self.image_processor.preprocess(np_video, return_tensors="pt")["pixel_values"]
                    videos.append(pixel_values)
                pixel_values = torch.stack(videos, dim=0)
        return BatchFeature(data={"input_ids": input_ids, "attention_mask": attention_mask, "images": pixel_values, "modalities": mode, "image_sizes": image_sizes})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)