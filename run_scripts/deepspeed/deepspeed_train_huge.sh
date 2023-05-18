
# ==== 基本参数 ====
num_gpus=3
PATH_D_INVIG=/mnt/bn/hri-lq/projects/VLDD/OFA-Invig
PATH_D_OFA=/mnt/bn/hri-lq/projects/VLDD/OFA
PATH_D_LOG=/mnt/bn/ckpt-lq/vldd

# ==== 运行训练 ====
cd ${PATH_D_INVIG}/ofa_invig
# deepspeed --num_gpus=${num_gpus} --master_port=2345 train_hf.py --deepspeed ${PATH_D_INVIG}/config/ds_config.json
python3 -m torch.distributed.run --nproc_per_node=${num_gpus} train_hf.py --deepspeed ds_config_fp16.json
