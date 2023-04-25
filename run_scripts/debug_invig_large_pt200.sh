#!/usr/bin/env

# ==== 基本参数 ====
num_gpus=1
PATH_D_INVIG=/mnt/bn/hri-lq/projects/VLDD/OFA-Invig
PATH_D_OFA=/mnt/bn/hri-lq/projects/VLDD/OFA
PATH_D_LOG=/mnt/bn/ckpt-lq/vldd

# ==== 预训练模型 ====
restore_file=/mnt/bn/hri-lq/projects/VLDD/OFA-checkpoints/ofa_large.pt
# restore_file=/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_2e-5_512_20230417-1746/checkpoint_best.pt
# restore_file=/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints_debug/10_1e-5_512_20230418-1555/checkpoint_last.pt
restore_file=/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230419-1954/checkpoint_last.pt
restore_file=/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_last.pt  # best?

# ==== 训练数据 ====
data="invig,invig"
selected_cols=0
python3 -c "from g2p_en import G2p"

# ==== 日志参数 ====
log_dir=${PATH_D_LOG}/invig_large_grounding_logs_debug
save_dir=${PATH_D_LOG}/invig_large_grounding_checkpoints_debug
mkdir -p $log_dir $save_dir
bpe_dir=${PATH_D_OFA}/utils/BPE
user_dir=${PATH_D_INVIG}/ofa_invig
cd $user_dir

# ==== 环境设置(无需更改) ====
export PYTHONPATH=$PYTHONPATH:${PATH_D_OFA}/fairseq
export MASTER_PORT=6051

# ==== 模型参数 ====
task=invig
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
warmup_ratio=0.06
batch_size=2
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=280
max_tgt_length=80
num_bins=1000
# lr=3e-5
# max_epoch=5
# patch_image_size=512

subfix=`date "+%Y%m%d-%H%M"`

for max_epoch in 10; do
  echo "max_epoch "${max_epoch}
  for lr in 3e-5; do
    echo "lr "${lr}
    for patch_image_size in 512; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}
      mkdir -p $save_path
      echo "log_file "${log_file}

      # torchrun --nnodes=1 --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} ${PATH_D_OFA}/train.py \
      # python3 -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} ${PATH_D_OFA}/train.py \
      python3 -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} --use_env ${PATH_D_OFA}/train.py \
          $data \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --restore-file=${restore_file} \
          --reset-optimizer --reset-dataloader --reset-meters \
          --save-dir=${save_path} \
          --task=${task} \
          --arch=${arch} \
          --criterion=${criterion} \
          --label-smoothing=${label_smoothing} \
          --batch-size=${batch_size} \
          --update-freq=${update_freq} \
          --encoder-normalize-before \
          --decoder-normalize-before \
          --share-decoder-input-output-embed \
          --share-all-embeddings \
          --layernorm-embedding \
          --patch-layernorm-embedding \
          --code-layernorm-embedding \
          --resnet-drop-path-rate=${resnet_drop_path_rate} \
          --encoder-drop-path-rate=${encoder_drop_path_rate} \
          --decoder-drop-path-rate=${decoder_drop_path_rate} \
          --dropout=${dropout} \
          --attention-dropout=${attention_dropout} \
          --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay --lr=${lr} \
          --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
          --log-format=simple --log-interval=10 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=1000 --validate-interval-updates=1000 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":1,"max_len_a":0,"max_len_b":40}' \
          --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --eval-print-samples \
          --num-workers=4 > ${log_file} 2>&1
    done
  done
done