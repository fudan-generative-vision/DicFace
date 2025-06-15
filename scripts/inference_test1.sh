cd /sykj_002/workspaces/HanlinShang/FVR_Project/FVR_OpenSource
source ../../miniconda3/bin/activate
conda activate fvr

ckpt_path=./ckpts/CodeFormer/paper_2_net_g_86000.pth
video_dir=/sykj_002/workspaces/HanlinShang/FVR_Project/LR_Blind
output_dir=/sykj_002/workspaces/HanlinShang/FVR_Project/FVR_OpenSource/test_output1/

mkdir -p "$output_dir"


for file in ${video_dir}/*; do

	filename=${file##*/}
	echo ${file}
	echo ${output_dir}${filename}
	echo "  "
	
	python scripts/inference_ours.py \
		-i ${file} \
		-o ${output_dir} \
		--max_length 10 \
		--save_video_fps 24 \
		--ckpt_path ${ckpt_path} \
		--bg_upsampler realesrgan \
		--save_video
	exit 1
done