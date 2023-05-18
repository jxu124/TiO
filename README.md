


### Docs
- https://fairseq.readthedocs.io/en/latest/
### build_model()
- OFA/models/ofa/unify_transformer.py:377

结论：ofa将图片首先预处理resize成固定的大小（根据modelsize决定，例如huge最大是512x512），然后用不同大小的[resnet](https://zhuanlan.zhihu.com/p/79378841)进行embedding，但是只用了前面几层，这里可以对照`OFA/models/ofa/resnet.py`这个文件看。最后输出了一个向量(N, 1024, 32, 32)(每个数字来源：512/2/2/2/2=32) 变形为(N, 32x32, 1024)。在用一个image_proj，把输出的1024维度转换为word_embed大小相等的(例如huge是1280)。最终和word_embed拼起来：[img_embed(1024), word_embed(~102)]


python3 src/bench_oracle_guesswhat.py -c "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"
python3 src/bench_grounding_guesswhat.py -c "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"
/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint_3_3000.pt
/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint_9_10400.pt


python3 ofa_invig/bench_grounding.py \
    --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint_9_10400.pt"\
    --batch_size 16\
    --task guesswhat_grounding


### 指令
评测：
```
python3 ofa_invig/bench_grounding.py \
    --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230428-1407/checkpoint_best.pt"\
    --batch_size 16\
    --task invig_grounding

python3 ofa_invig/bench_grounding.py \
    --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint_3_3000.pt"\
    --batch_size 32\
    --task guesswhat_grounding

python3 ofa_invig/bench_oracle_guesswhat.py \
    --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230428-1407/checkpoint_best.pt"\
    --batch_size 16
```

性能：
| model | guesswhat_grounding | invig_grounding | guesswhat_oracle |
| - | - | - | - |
| /mnt/bn/hri-lq/projects/VLDD/OFA-Invig/results-debug/checkpoint-60500/pytorch_model.bin | 0.721 | 0.758 | - |
| /mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230428-1407/checkpoint_best.pt | 0.0 | 0.767 | - |

迭代2
```


```


迭代1
```
#instruction: human: 
#context:
#region:

其中：AI以agent自称，人类为huamn
```

数据集features:
global_image_id - string
bbox - list
image_path - string
captions - list
caption - string
dialog - list
dialog_cn - list

id - string
category - string
raw_* - string


### 评估指令
```bash
### invig_grounding
# 1 & 2 & 3 & 4
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "invig-large" --ckpt 
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "invig-m-task" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230509-2110/checkpoint_last.pt"
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "invig-m-grounding" --ckpt 
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "invig-m-dialog" --ckpt 
# 5 & 7
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "large-o-invig" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230505-1510/checkpoint_last.pt"
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "large-o-grounding+invig" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230511-0146/checkpoint_last.pt"

# 1 
# 2 14254328 /mnt/bn/ckpt-lq/vldd/benchmark/invig_grounding/invig-m-task/20230512-163639.json
# 3 
# 4 
# 5 14255137 /mnt/bn/ckpt-lq/vldd/benchmark/invig_grounding/large-o-invig/20230512-172315.json
# 7 14255129 /mnt/bn/ckpt-lq/vldd/benchmark/invig_grounding/large-o-grounding+invig/20230512-171030.json

### guesswhat_grounding
# 1 & 2 & 3 & 4
# optional

# 6 & 8
python3 ofa_invig/bench_on_dataset.py --task guesswhat_grounding --comment "large-o-guesswhat" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230510-1719/checkpoint_last.pt"
python3 ofa_invig/bench_on_dataset.py --task guesswhat_grounding --comment "large-o-grounding+guesswhat" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230511-2231/checkpoint_last.pt"

# 1
# 2
# 3
# 4
# 6 /mnt/bn/ckpt-lq/vldd/benchmark/guesswhat_grounding/large-o-guesswhat/20230512-160710.json
# 8 


### guesswhat_oracle
# 1 & 2 & 3 & 4

# 6 & 8
python3 ofa_invig/bench_on_dataset.py --task guesswhat_oracle --comment "large-o-guesswhat" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230510-1719/checkpoint_last.pt"
python3 ofa_invig/bench_on_dataset.py --task guesswhat_oracle --comment "large-o-grounding+guesswhat" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230511-2231/checkpoint_last.pt"


### invig_grounding_end2end
python3 ofa_invig/bench_on_dataset_debug.py --task invig_grounding_end2end --comment "debug" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230511-0146/checkpoint.best_score_0.7361.pt"


### guesswhat_grounding_end2end



### Huge main
invig_grounding invig_grounding_end2end
guesswhat_grounding guesswhat_grounding_end2end guesswhat_oracle
# /mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "huge-main-0.7630" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt"
invig_grounding - 14256730 /mnt/bn/ckpt-lq/vldd/benchmark/invig_grounding/huge-main-0.7630/20230512-181257.json
invig_grounding_end2end - 
guesswhat_grounding - 
guesswhat_grounding_end2end - 
guesswhat_oracle - 

# /mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint.best_score_0.7740.pt
python3 ofa_invig/bench_on_dataset.py --task invig_grounding --comment "huge-main-0.7740" --ckpt "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_1e-5_512_20230504-0049/checkpoint.best_score_0.7740.pt"
invig_grounding - 14256745 /mnt/bn/ckpt-lq/vldd/benchmark/invig_grounding/huge-main-0.7740/20230512-182551.json
invig_grounding_end2end - 
guesswhat_grounding - 
guesswhat_grounding_end2end - 
guesswhat_oracle - 



```

