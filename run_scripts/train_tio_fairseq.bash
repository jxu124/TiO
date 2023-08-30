#!/bin/bash
python3 -c "from g2p_en import G2p"
sudo pip3 install -U transformers > /dev/null

# ***** 超参数 *****
num_gpus=$(python3 -c "import torch;print(torch.cuda.device_count())")
data="invig,invig"
# max_epoch=10
# lr=3e-5
# warmup_ratio=0.06
max_epoch=1
lr=1e-5
warmup_ratio=0.3
batch_size=4
update_freq=4
patch_image_size=512
restore_file=./attachments/ofa_huge.pt
restore_file=./attachments/checkpoint.pt


# ***** 路径配置 *****
log_dir=/mnt/bn/ckpt-lq/ckpt/tio/tio_logs
save_dir=/mnt/bn/ckpt-lq/ckpt/tio/tio_ckpts
# 自动计算其他路径
mkdir -p $log_dir $save_dir
work_dir=$(readlink -f $(dirname $(dirname "$0")))
module_ofa=${work_dir}/attachments/OFA
user_dir=${work_dir}/tio_core
config_yaml=${work_dir}/config/training.yml
bpe_dir=${module_ofa}/utils/BPE
# export PYTHONPATH=${module_ofa}/fairseq:$PYTHONPATH
cd ${work_dir}

# ***** 修正OFA的一些bug *****
# sed -i 's/np.float/np.float32/g' ${module_ofa}/fairseq/fairseq/data/indexed_dataset.py

# ***** 超参数（不需要动） *****
task=invig
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=240
max_tgt_length=280
num_bins=1000


# ***** 打印必要内容 *****
echo "work_dir "$work_dir
echo "num_gpus "$num_gpus
echo "max_epoch "$max_epoch
echo "lr "${lr}
echo "patch_image_size "${patch_image_size}

subfix=`date "+%Y%m%d-%H%M"`
log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_"${subfix}
mkdir -p $save_path
echo "log_file: "${log_file}


# ***** 运行 *****
pt_version=$(python3 -c "import torch;print(torch.__version__)")
echo $pt_version
if [[ ${pt_version} == "1"* ]]; then
    launcher="python3 -m torch.distributed.launch --master_port 2345"
else
    launcher="torchrun --master-port 2345"
fi
${launcher} --nproc_per_node ${num_gpus} ${module_ofa}/train.py \
  $data \
  --config-yaml=${config_yaml} \
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
  --save-interval-updates=1000 --validate-interval-updates=1000 \
  --eval-acc \
  --eval-args='{"beam":5,"min_len":1,"max_len_a":0,"max_len_b":20}' \
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