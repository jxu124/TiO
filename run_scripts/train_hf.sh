
cd /mnt/bn/hri-lq/projects/VLDD/OFA-Invig
#  --fsdp "shard_grad_op offload auto_wrap"
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6402 src/train_hf.py > training.log 2>&1

