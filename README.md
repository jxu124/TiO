


### Docs
- https://fairseq.readthedocs.io/en/latest/
### build_model()
- OFA/models/ofa/unify_transformer.py:377

结论：ofa将图片首先预处理resize成固定的大小（根据modelsize决定，例如huge最大是512x512），然后用不同大小的[resnet](https://zhuanlan.zhihu.com/p/79378841)进行embedding，但是只用了前面几层，这里可以对照`OFA/models/ofa/resnet.py`这个文件看。最后输出了一个向量(N, 1024, 32, 32)(每个数字来源：512/2/2/2/2=32) 变形为(N, 32x32, 1024)。在用一个image_proj，把输出的1024维度转换为word_embed大小相等的(例如huge是1280)。最终和word_embed拼起来：[img_embed(1024), word_embed(~102)]


python3 src/bench_oracle_guesswhat.py -c "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"
python3 src/bench_grounding_guesswhat.py -c "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"

### 指令
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

