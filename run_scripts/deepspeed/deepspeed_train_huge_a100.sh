
# ==== 基本参数 ====
num_gpus=8
batch_size=9
checkpoint=/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230427-1312/checkpoint_1_2000.pt
checkpoint=/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt
checkpoint=/mnt/bn/ckpt-lq/vldd/invig_huge_checkpoints/10_3e-5_512_20230424-2228/checkpoint_last.pt
PATH_D_INVIG=/mnt/bn/hri-lq/projects/VLDD/OFA-Invig
PATH_D_OFA=/mnt/bn/hri-lq/projects/VLDD/OFA
PATH_D_LOG=/mnt/bn/ckpt-lq/vldd

# ==== 运行训练 ====
cd ${PATH_D_INVIG}/ofa_invig
# deepspeed --num_gpus=${num_gpus} --master_port=2345 train_hf.py --deepspeed ${PATH_D_INVIG}/config/ds_config.json
python3 -m torch.distributed.run --nproc_per_node=${num_gpus} --master_port=2345 train_hf.py \
    --checkpoint $checkpoint \
    --batch_size $batch_size \
    --bf16
    # --deepspeed ${PATH_D_INVIG}/config/ds_config_bf16_stage2.json  # 注释掉以使用 fsdp
