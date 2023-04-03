#!/usr/bin/env

# ==== 基本参数 ====
num_gpus=8
# load from .env file
ENV_FILE=$(dirname "$0")/../.env
export $(xargs < $ENV_FILE)

# ==== 训练数据 ====
data=${PATH_D_DATASET}/invig_train.jsonl,${PATH_D_DATASET}/guesswhat_train.jsonl,${PATH_D_DATASET}/visdial_train.jsonl,${PATH_D_DATASET}/invig_valid.jsonl
restore_file=${PATH_D_CHECKPOINTS}/ofa_large.pt
selected_cols=0

# ==== 日志参数 ====
log_dir=${PATH_D_LOG}/invig_test_logs
save_dir=${PATH_D_LOG}/invig_test_checkpoints
mkdir -p $log_dir $save_dir
bpe_dir=${PATH_D_OFA}/utils/BPE
user_dir=${PATH_D_INVIG}/invig_module

# ==== 环境设置(无需更改) ====
export PYTHONPATH=$PYTHONPATH:${PATH_D_OFA}/fairseq
export MASTER_PORT=6051

# ==== 模型参数 ====
task=invig
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
warmup_ratio=0.06
batch_size=8
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=300
max_tgt_length=100
num_bins=1000
# lr=3e-5
# max_epoch=5
# patch_image_size=512

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

subfix=`date "+%Y%m%d-%H%M"`

for max_epoch in 20; do
  echo "max_epoch "${max_epoch}
  for lr in 3e-5; do
    echo "lr "${lr}
    for patch_image_size in 512; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}
      mkdir -p $save_path
      echo "log_file "${log_file}

      # python3 -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} 
      python3 ${PATH_D_OFA}/train.py \
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
          --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
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
          ${uses_ema} \
          ${store_ema} \
          ${ema_fp32} \
          --ema-decay=${ema_decay} \
          --ema-start-update=${ema_start_update} \
          --num-workers=0
    done
  done
done