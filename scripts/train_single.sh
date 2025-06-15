cd /sykj_002/workspaces/HanlinShang/FVR_Project/FVR_OpenSource
source /sykj_002/workspaces/HanlinShang/miniconda3/bin/activate

conda activate codeformer

CUDA_VISIBLE_DEVICES=0 \
python train.py -opt configs/test_config.yaml