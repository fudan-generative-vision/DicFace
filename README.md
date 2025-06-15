# DVFR

## install 

```
# git clone this repository
git clone 
cd 

# create new anaconda env
conda create -n fvr python=3.10 -y
conda activate fvr

# install pytorch
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib (only for face detection or cropping with dlib)
```

## inference

```
# inference scripts
python scripts/inference_ours.py \
		-i /path/to/video \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path  \
		--bg_upsampler realesrgan \
		--save_video

```

### for blind face restoration
```
python scripts/inference_ours.py \
		-i /path/to/video \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path  \
		--bg_upsampler realesrgan \
		--save_video 

# or your videos has been aligned
python scripts/inference_ours.py \
		-i /path/to/video \
		-o /path/to/output_folder \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path  \
		--bg_upsampler realesrgan \
		--save_video \
		--has_aligned
```