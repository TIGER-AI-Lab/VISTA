# VISTA

This repo contains code for [VISTA](https://arxiv.org/abs/2412.00927), a video spatiotemporal augmentation method that generates long-duration and high-resolution video instruction-following data to enhance the video understanding capabilities of video LMMs.

### This repo is under construction. Please stay tuned.
[**🌐 Homepage**](https://tiger-ai-lab.github.io/VISTA/) | [**📖 arXiv**](https://arxiv.org/abs/2412.00927) | [**💻 GitHub**](https://github.com/TIGER-AI-Lab/VISTA) | [**🤗 VISTA-400K**](https://huggingface.co/datasets/TIGER-Lab/VISTA-400K) | [**🤗 Models**](https://huggingface.co/collections/TIGER-Lab/vista-674a2f0fab81be728a673193) | [**🤗 HRVideoBench**](https://huggingface.co/datasets/TIGER-Lab/HRVideoBench)

## Video Instruction Data Synthesis Pipeline
<p align="center">
<img src="https://tiger-ai-lab.github.io/VISTA/static/images/vista_main.png" width="900">
</p>

VISTA leverages insights from image and video classification data augmentation techniques such as CutMix, MixUp and VideoMix, which demonstrate that training on synthetic data created by overlaying or mixing multiple images or videos results in more robust classifiers. Similarly, our method spatially and temporally combines videos to create (artificial) augmented video samples with longer durations and higher resolutions, followed by synthesizing instruction data based on these new videos. Our data synthesis pipeline utilizes existing public video-caption datasets, making it fully open-sourced and scalable. This allows us to construct VISTA-400K, a high-quality video instruction-following dataset aimed at improving the long and high-resolution video understanding capabilities of video LMMs.



## Citation
If you find our paper useful, please cite us with
```
@misc{ren2024vistaenhancinglongdurationhighresolution,
      title={VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation}, 
      author={Weiming Ren and Huan Yang and Jie Min and Cong Wei and Wenhu Chen},
      year={2024},
      eprint={2412.00927},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00927}, 
}
```
