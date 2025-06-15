cd /sykj_002/workspaces/HanlinShang/FVR_Project/FVR_OpenSource
source ../../miniconda3/bin/activate
conda activate fvr

ckpt_path=/sykj_002/workspaces/HanlinShang/FVR_Project/FVR_project/experiments/20250531_093643_codeformer_dirichlet_clip5_bs2_align_nofix_multiscale_multi_gpus/models/net_g_100000.pth
video_dir=/sykj_002/workspaces/HanlinShang/FVR_Project/LR_Blind
output_dir=/sykj_002/workspaces/HanlinShang/FVR_Project/FVR_OpenSource/test_output2/open_bfr_100k_no_dloss_origin_test_data

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
	# exit 1
done