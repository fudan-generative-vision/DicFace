<h1 align='center'>DicFace: Dirichlet-Constrained Variational Codebook Learning for Temporally Coherent Video Face Restoration</h1>

<div align='center'>
    <a href='' target='_blank'>Yan Chen</a><sup>1*</sup>&emsp;
    <a href='' target='_blank'>Hanlin Shang</a><sup>1*</sup>&emsp;
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
    <a href='https://arxiv.org/abs/2506.13355'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> -->
    <!-- <a href='assets/wechat.jpeg'><img src='https://badges.aleen42.com/src/wechat.svg'></a> -->
</div>

<br>

<table align="center" border="0" style="width: 100%; margin-top: 80px;">
  <tr>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/274ecc2b-3d89-4d31-bb0a-a5f3611fae8a" 
             muted autoplay loop style="display: block; margin: 0 auto;"></video>
    </td>
  </tr>
</table>
## 📸 Showcase

### Blind Face Restoration
<table align="center" width="100%" border="0" cellpadding="10">
  <tr>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/eb61d793-b860-476e-bae5-f6fcade1e11f" muted autoplay loop width="480"></video>
    </td>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/eb9be43a-8fb9-4fbd-ac92-a686ab0c188b" muted autoplay loop width="480"></video>
    </td>
  </tr>
</table>


### Face Inpainting
<table align="center" width="100%" border="0" cellpadding="10">
  <tr>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/1cd12d53-2ead-4cf3-b56c-1a6316484e93" muted autoplay loop width="480"></video>
    </td>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/a16b7021-a401-41cb-9a39-37a788f6a001" muted autoplay loop width="480"></video>
    </td>
  </tr>
</table>

### Face Colorization
<table align="center" width="100%" border="0" cellpadding="10">
  <tr>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/cb038911-8b26-472d-8fb9-a6cdda127084" muted autoplay loop width="480"></video>
    </td>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/ffc85ef7-4987-42af-b892-79544ea29f87" muted autoplay loop width="480"></video>
    </td>
  </tr>
</table>

## 📰 News

- **`2025/06/26`**: 🎉🎉🎉 Our paper has been accepted to [ICCV 2025](https://iccv.thecvf.com/Conferences/2025).
- **`2025/06/25`**: Release our test data on huggingface [repo](https://huggingface.co/datasets/fudan-generative-ai/DicFace-test_dataset).
- **`2025/06/23`**: Release our pretrained model on huggingface [repo](https://huggingface.co/fudan-generative-ai/DicFace).
- **`2025/06/17`**: Paper submitted on Arixiv. [paper](https://arxiv.org/abs/2506.13355)
- **`2025/06/16`**: 🎉🎉🎉 Release inference scripts



## 📅️ Roadmap

| Status | Milestone                                                                                              |    ETA     |
| :----: | :----------------------------------------------------------------------------------------------------- | :--------: |
|   ✅   | **[Inference Code release](https://github.com/fudan-generative-vision/DicFace)**                       |  2025-6-16 |
|   ✅   | **[Model Weight release， baidu-link](https://pan.baidu.com/s/1VTNbdtZDvgY0163a1T8ITw?pwd=dicf)**       |2025-6-16   |
|   ✅   | **[Paper submitted on Arixiv](https://arxiv.org/abs/2506.13355)**                                       |  2025-6-17 |
|   ✅   | **[Test data release](https://huggingface.co/datasets/fudan-generative-ai/DicFace-test_dataset)**       |  2025-6-25 |
|   🚀   | **[Training Code release]()**                                                                           |  2025-6-28 |



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

The pre-trained weights have been uploaded to Baidu Netdisk. Please download them from the [link](https://pan.baidu.com/s/1VTNbdtZDvgY0163a1T8ITw?pwd=dicf)

Now you can easily get all pretrained models required by inference from our HuggingFace [repo](https://huggingface.co/fudan-generative-ai/DicFace_model).

**File Structure of Pretrained Models**
The downloaded .ckpts directory contains the following pre-trained models:

```
.ckpts
|-- CodeFormer                  # CodeFormer-related models
|   |-- bfr_100k.pth            # Blind Face Restoration model 
|   |-- color_100k.pth          # Color Restoration model 
|   `-- inpainting_100k.pth     # Image Inpainting model
|-- dlib                        # dlib face-related models
|   |-- mmod_human_face_detector.dat  # Human face detector
|   `-- shape_predictor_5_face_landmarks.dat  # 5-point face landmark predictor
|-- facelib                     # Face processing library models
|   |-- detection_Resnet50_Final.pth  # ResNet50 face detector 
|   |-- detection_mobilenet0.25_Final.pth  # MobileNet0.25 face detector 
|   |-- parsing_parsenet.pth    # Face parsing model
|   |-- yolov5l-face.pth        # YOLOv5l face detection model
|   `-- yolov5n-face.pth        # YOLOv5n face detection model
|-- realesrgan                  # Real-ESRGAN super-resolution model
|   `-- RealESRGAN_x2plus.pth   # 2x super-resolution enhancement model
`-- vgg                         # VGG feature extraction model
    `-- vgg.pth                 # VGG network pre-trained weights
```

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

## Test Data  

Our test data can be accessed via the following links:  
- Baidu Netdisk: [https://pan.baidu.com/s/1zMp3fnf6LvlRT9CAoL1OUw](https://pan.baidu.com/s/1zMp3fnf6LvlRT9CAoL1OUw) (Password: `drhh`)  
- Hugging Face Dataset: [https://huggingface.co/datasets/fudan-generative-ai/DicFace-test_dataset](https://huggingface.co/datasets/fudan-generative-ai/DicFace-test_dataset)  


### Directory Structure  
The downloaded `test_data_set` directory contains the following folders:  
```
./test_data
├── LR_Blind                  # Blind face restoration test image folders
│   ├── Clip+_HebIzK_LP4+P2+C1+F16589-16715
│   ├── ...                   # Additional test image folders
│   └── Clip+y5OFsRIRkwc+P0+C0+F9797-9938
│
├── TEST_DATA                 # Ground-truth (GT) image folders
│   ├── Clip+_HebIzK_LP4+P2+C1+F16589-16715
│   ├── ...
│   └── Clip+y5OFsRIRkwc+P0+C0+F9797-9938
│
├── vfhq_test_color_input     # Colorization test image folders
│   ├── Clip+_HebIzK_LP4+P2+C1+F16589-16715
│   ├── ...
│   └── Clip+y5OFsRIRkwc+P0+C0+F9797-9938
│
├── vfhq_test_inpaint_input_512  # Inpainting test image folders (512x512)
│   ├── Clip+_HebIzK_LP4+P2+C1+F16589-16715
│   ├── ...
│   └── Clip+y5OFsRIRkwc+P0+C0+F9797-9938
│
└── vfhq_test_landmarks       # Facial landmark files for warping operations
```


### Usage  
To process the test data, use the `warp_images.py` script:  
```shell
python scripts/warp_images.py \
    -i input_test_data_folder \        # Input folder containing test data
    -o vfhq_test_inpaint_input_512_warped \  # Output folder for warped results
    -l /path/to/test_data_folder/vfhq_test_landmarks  # Landmark file directory
```  

After warping the test data, you can use the inference scripts to generate results for the test dataset.


## Training

#### Training data



## 🤗 Acknowledgements

This project is open sourced under NTU S-Lab License 1.0. Redistribution and use should follow this license. The code framework is mainly modified from [CodeFormer](https://github.com/sczhou/CodeFormer). Please refer to the original repo for more usage and documents.

## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{chen2025dicfacedirichletconstrainedvariationalcodebook,
      title={DicFace: Dirichlet-Constrained Variational Codebook Learning for Temporally Coherent Video Face Restoration}, 
      author={Yan Chen and Hanlin Shang and Ce Liu and Yuxuan Chen and Hui Li and Weihao Yuan and Hao Zhu and Zilong Dong and Siyu Zhu},
      year={2025},
      eprint={2506.13355},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.13355}, 
}

```

