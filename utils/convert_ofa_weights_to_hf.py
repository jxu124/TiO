import os
import torch

# 这是一个DEBUG version from ofa's github issue.

# ==== 获取模型参数模板 ====
# https://huggingface.co/OFA-Sys/OFA-huge
# https://huggingface.co/OFA-Sys/OFA-large

# 1 安装 ofa transformer ver.
# !cd /root/data/checkpoints && unzip OFA-feature-add_transformers.zip
# !cd /root/data/checkpoints/OFA-feature-add_transformers && pip3 install ./transformers
# git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
# pip install OFA/transformers/

# 2 获取模型参数模板
# !cd /root/data/checkpoints && git clone https://huggingface.co/OFA-Sys/OFA-large
# !cd /root/data/checkpoints && git clone https://huggingface.co/OFA-Sys/OFA-huge

# ==== 参数设置 ====
# 模板模型文件夹位置
template_model = "/root/data/checkpoints/OFA-large"
# 需要被转换的fairseq模型位置
source_model = '/root/data/checkpoints/ofa/ofa_large.pt'
source_model = '/mnt/bn/hri-lq/projects/OFA/logs/invig_checkpoints/10_3e-5_512_20230321-1827/checkpoint_10_38500.pt'
# 需要新建transformers模型的路径文件夹
target_model = "/root/data/projects/OFA/test/eval-invig-model-large"
# target_model = "/root/data/projects/OFA/test/eval-invig-2000-large"

# 转换模型参数 （converting OFA fairseq ckpt --> huggingface inference ckpt）
# https://colab.research.google.com/drive/1LLJewY92LXdeug5m_ceMUHdlqrRQwSQJ?usp=sharing#scrollTo=KMkxZWEY_EHE
# fairseq to transformers
# load the HF and Fairseq variants to mm1 and m2
# note: Fairseq variant also stores optimizer history and and some other training-related information
# in other keys, alongside the model parameters
# in the key 'model'.
# we only need it.
import torch

# m1 是要保存参数的位置（transformers）
# m2 是fairseq方式训练得到的模型参数
m1 = torch.load(f'{template_model}/pytorch_model.bin')
m2 = torch.load(source_model)['model']
# m2 = torch.load('')['model']

# ----------------
# store parameter names in Python sets
# so that we can get the intersection easily
# and do some staff with O(1) efficiency later on.
k1 = set([k for k in m1.keys()])
k2 = set([k for k in m2.keys()])
len(k1), len(k2)

# ----------------
# how many parameters share the a common name
common_k = k1.intersection(k2)
print(len(common_k))

# ----------------
for k in m1.keys():
    if k in common_k:
        if m1[k].shape != m2[k].shape:
            print(k, m1[k].shape, m2[k].shape)
        m1[k] = m2[k]
        del m2[k]
        k1.remove(k)
        k2.remove(k)
print(len(k1), len(k2))

# ----------------
# we discovered the following changes in naming
# *.ffn_layernorm -> *.ffn_layer_norm
# *.self_attn_ln.* -> *.self_attn_mid_layer_norm.*
# *.cross_attn_ln.* -> *.cross_attn_mid_layer_norm.*
# *.encoder_attn.* -> *.cross_attn.*
# 
# we will do make these replacements, and if such a key actually exists
# in the HF variant, we will update it with the parameter
# from the Fairseq variant

for k in m2.keys():
    k_pred = k.replace('ffn_layernorm', 'ffn_layer_norm')
    k_pred = k_pred.replace('self_attn_ln', 'self_attn_mid_layer_norm')
    k_pred = k_pred.replace('cross_attn_ln', 'cross_attn_mid_layer_norm')
    k_pred = k_pred.replace('encoder_attn', 'cross_attn')
    if k_pred in k1:
        if m1[k_pred].shape != m2[k].shape:
            print(k, k_pred, m1[k_pred].shape, m2[k].shape)
        m1[k_pred] = m2[k]
        k1.remove(k_pred)
        k2.remove(k)
print(len(k1), len(k2))

# ----------------
# there's only one mapping for the remaining parameters:
# *.attn_ln.* -> *.self_attn_mid_layer_norm.*

for k in m2.keys():
    k_pred = k.replace('attn_ln', 'self_attn_mid_layer_norm')
    if k_pred in k1:
        if m1[k_pred].shape != m2[k].shape:
            print(k, k_pred, m1[k_pred].shape, m2[k].shape)
        m1[k_pred] = m2[k]
        k1.remove(k_pred)
        k2.remove(k)

print(len(k1), len(k2))
print(k2)

# ----------------
# 现在 /root/data/projects/OFA/checkpoints/eval 中保存了ofa-large的模型参数
os.system(f"rm -r {target_model}")
os.system(f"mkdir -p {target_model}")
os.system(f"cp -r {template_model}/* {target_model}")
torch.save(m1, f'{target_model}/pytorch_model.bin')
