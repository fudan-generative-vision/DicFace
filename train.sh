CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 --master_port=29597 \
    basicsr/train.py \
    -opt options/clip5_bs2_512_align_nofix_multiscale.yaml \
    --launcher pytorch
