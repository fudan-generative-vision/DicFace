<h1 align='center'>DicFace: Dirichlet-Constrained Variational Codebook Learning for Temporally Coherent Video Face Restoration</h1>

<div align='center'>
    <a href='' target='_blank'>Yan Chen</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Hanlin Shang</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Ce Liu</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Yuxuan Chen</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Hui Li</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Weihao Yuan</a><sup>2</sup>&emsp;
</div>
<div align='center'>
    <a href='' target='_blank'>Hao Zhu</a><sup>3</sup>&emsp;
    <a href='' target='_blank'>Zilong Dong</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>1✉️</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Fudan University&emsp; 
    <sup>2</sup>Alibaba Group&emsp;
    <sup>3</sup>Nanjing University&emsp;
</div>

<br>
<div align='center'>
    <a href='https://github.com/fudan-generative-vision/DicFace'><img src='https://img.shields.io/github/stars/fudan-generative-vision/DicFace'></a>
    <!-- <a href='https://github.com/fudan-generative-vision/DicFace/#/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a> -->
    <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> -->
    <!-- <a href='assets/wechat.jpeg'><img src='https://badges.aleen42.com/src/wechat.svg'></a> -->
</div>
<!-- <div align='Center'>
    <i><strong><a href='https://cvpr.thecvf.com/Conferences/2025' target='_blank'>CVPR 2025</a></strong></i>
</div> -->
<br>

<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
  <tr>
    <td>
      <video align='center' src="https://github.com/user-attachments/assets/274ecc2b-3d89-4d31-bb0a-a5f3611fae8a" muted autoplay loop></video>
    </td>
  </tr>
</table>

## 📸 Showcase

### BFR Task
<video align='center' src="https://github.com/user-attachments/assets/63907f13-0921-4dd8-a074-dce818710d59" muted autoplay loop></video>

### Inpainting Task
<video align='center' src="https://github.com/user-attachments/assets/6c1aab2b-905f-4a93-acbc-e4a6b61233d9" muted autoplay loop></video>

### Colorization Task
<video align='center' src="https://github.com/user-attachments/assets/399b5474-d565-4616-81b9-95a1df014044" muted autoplay loop></video>

## 📅️ Roadmap

| Status | Milestone                                                                                    |    ETA     |
| :----: | :------------------------------------------------------------------------------------------- | :--------: |
|   🚀   | **[Paper submitted on Arixiv]()**                                                            |    TBD     |
|   🚀   | **[Inference Code release]()**                                                               |    TBD     |
|   🚀   | **[Test data release]()**                                                                    |    TBD     |
|   🚀   | **[Training Code release]()**                                                                |    TBD     |
|   🚀   | **[Training data release]()**                                                                |    TBD     |


## 📰 News

- **`2025/06/16`**: 🎉🎉🎉 Release inference scripts

## ⚙️ Installation

- System requirement: PyTorch version >=2.4.1, python == 3.10
- Tested on GPUs: A800, python version == 3.10, PyTorch version == 2.4.1, cuda version == 12.1

Download the codes:

```bash
  git clone https://github.com/fudan-generative-vision/DicFace
  cd DicFace
```

Create conda environment:

```bash
  conda create -n DicFace python=3.10
  conda activate DicFace
```

Install PyTorch

```bash
  conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
  python basicsr/setup.py develop
  conda install -c conda-forge dlib
```

### 📥 Download Pretrained Models


### 🛠️ Prepare Inference Data

### 🎮 Run Inference

#### for blind face restoration

```bash
python scripts/inference.py \
		-i /path/to/video \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path /bfr/bfr_weight.pth \
		--bg_upsampler realesrgan \
		--save_video 

# or your videos has been aligned
python scripts/inference.py \
		-i /path/to/video \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path /bfr/bfr_weight.pth \
		--bg_upsampler realesrgan \
		--save_video \
		--has_aligned
```

#### for colorization & inpainting task


**The current colorization & inpainting tasks only supports input of aligned faces. If a non-aligned face is input, it may lead to unsatisfactory final results.**

``` bash 
# for colorization task
python scripts/inference_color_and_inpainting.py \
		-i /path/to/video_warped \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path /colorization/colorization_weight.pth \
		--bg_upsampler realesrgan \
		--save_video \
		--has_aligned

# for inpainting task
python scripts/inference_color_and_inpainting.py \
		-i /path/to/video_warped \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path /inpainting/inpainting_weight.pth \
		--bg_upsampler realesrgan \
		--save_video \
		--has_aligned
```



## Training
**TBD**
## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

