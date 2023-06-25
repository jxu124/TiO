#!/usr/bin/env

# ====  ====
num_gpus=8
PATH_D_INVIG=
PATH_D_OFA=
PATH_D_LOG=

# ====  ====
config_yaml=./config/training.yml
restore_file=./attachments/pretrain.pt

# ====  ====
data="invig,invig"
selected_cols=0
python3 -c "from g2p_en import G2p"

# ====  ====
log_dir=${PATH_D_LOG}/tio_logs
save_dir=${PATH_D_LOG}/tio_checkpoints
mkdir -p $log_dir $save_dir
bpe_dir=${PATH_D_OFA}/utils/BPE
user_dir=${PATH_D_INVIG}/ofa_invig
cd $user_dir

# ====  ====
export PYTHONPATH=$PYTHONPATH:${PATH_D_OFA}/fairseq
export MASTER_PORT=6051

# ====  ====
task=invig
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
warmup_ratio=0.06
batch_size=5
update_freq=3
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=240
max_tgt_length=280
num_bins=1000

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

      python3 -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} ${PATH_D_OFA}/train.py \
          $data \
          --config-yaml=${config_yaml} \
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
          --no-epoch-checkpoints --keep-best-checkpoints=2 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=2000 --validate-interval-updates=1000 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":1,"max_len_a":0,"max_len_b":100}' \
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
          --num-workers=2 > ${log_file} 2>&1
    done
  done
done