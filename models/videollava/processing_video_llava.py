import numpy as np
from PIL import Image
from typing import List, Optional, Union
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.video_llava.processing_video_llava import VideoLlavaProcessor


class AdaptedVideoLlavaProcessor(VideoLlavaProcessor):
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
    
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        input_images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        do_resize: bool = False,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        videos = images = None
        if isinstance(input_images, Image.Image):
            input_images = [input_images]
        if isinstance(input_images, list) and isinstance(input_images[0], Image.Image):
            images = np.stack([np.array(frame) for frame in input_images])
        elif isinstance(input_images, list) and isinstance(input_images[0], list):
            videos = [np.stack([np.array(frame) for frame in clip]) for clip in input_images]

        return super().__call__(
            text=text,
            images=images,
            videos=videos,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )