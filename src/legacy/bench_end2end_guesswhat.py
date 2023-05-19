# 基本设置
import argparse
import yaml
args = argparse.Namespace(**{
    "path_ofa": "/mnt/bn/hri-lq/projects/VLDD/OFA",
    "path_invig": "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig",
    "path_tokenizer": "/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large"
})

# 路径设置
import copy
import os
import sys
import re
import json
import random
import time
sys.path.append(args.path_ofa)
os.chdir(args.path_invig)

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
from PIL import Image
# from dataclasses import dataclass
# from torch.cuda.amp import autocast
import logging
import numpy as np
import torch
import mmcv

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{args.path_invig}/src"}))

# 载入transformers
from src.ofa.tokenization_ofa import OFATokenizer
# from transformers.modeling_outputs import Seq2SeqLMOutput
# from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

        
# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.legacy.dialog_dataset import TaskProcessor, DataCollatorForOFA, OFADataset
from src.legacy.dialog_dataset import LoadImage
from src.common import world_info_from_env, get_image_processor, sbbox_to_bbox, bbox_to_sbbox
from src.common import sbbox_to_bbox
from src.evaluation import eval_grounding_acc_v2
from src.common import OFAModelWarper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

with open('/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_4ds.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def make_a_dialog(data, model):
    # args: data, model

    dialog, sbbox_target, image = data
    image = image.convert("RGB")
    bbox = sbbox_to_bbox(sbbox_target, *image.size)
    sbbox = bbox_to_sbbox(bbox, *image.size)

    # 第一轮：人类提出query
    # src_text = f" \n#instruction: say a word about the region. \n#region: {sbbox}"
    # gen_text = model.generate([src_text], [image])[0].strip()
    # context = f"query: {gen_text}"
    # all_dialogs = [("", gen_text)]
    # print(f"[Turn 1] query: {gen_text}")
    # logs = f"[Turn 1] query: {gen_text}\n"
    context = "query: guess what i want."
    all_dialogs = []
    logs = ""

    # 第二轮之后：循环直到给出bbox
    dialog_loop = True
    for turn in range(1, 10+1):  # max_turn = 10
        # 提问
        if turn >= 10:  # 强行结束
            src_text = f" \n#instruction: which region does the context describe? \n#context: \"{context}\""
        elif turn >= 5:
            src_text = f" \n#instruction: can you specify which region the context describes? \n#context: \"{context}\""
        else:
            src_text = f" \n#instruction: ask a question. \n#context: \"{context}\""
        gen_text = model.generate([src_text], [image])[0].strip()
        question = re.sub("not yet\.", "", gen_text).strip()
        context = context

        # 如果给出了bbox，就跳出循环
        if gen_text.startswith("sure.") or gen_text.startswith("region:"):
            pred_bbox = sbbox_to_bbox(gen_text, *image.size)
            break

        # 回答
        src_text = f" \n#instruction: answer the question with yes or no. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\""
        # src_text = f" \n#instruction: answer the question in detail. \n#region: {sbbox}\n#context: \"{context}\"\n#question: \"{question}\""
        gen_text = model.generate([src_text], [image])[0].strip()
        answer = gen_text
        context = context + f" question: {question} answer: {answer}"

        # 添加对话历史
        all_dialogs += [(question, answer)]
        print(f"[Turn {turn}] question: {question} answer: {answer}")
        logs += f"[Turn {turn}] question: {question} answer: {answer}\n"

    bboxes = bbox[None, :]
    pred_bboxes = pred_bbox[None, :]
    status = eval_grounding_acc_v2(pred_bboxes, bboxes)['grounding_acc_iou_0.5'] > 0.5
    print(f"[Turn {turn}] grounding: {pred_bbox} ({'success' if status else 'failure'})")
    logs += f"[Turn {turn}] grounding: {pred_bbox} ({'success' if status else 'failure'})\n"

    bbox_image = mmcv.imshow_bboxes(np.asarray(image)[:, :, ::-1], [bboxes, pred_bboxes], colors=['green', 'red'], show=False)[:, :, ::-1]
    ret = {
        "status": status,
        "dialog": all_dialogs,
        "text": logs,
        "image": Image.fromarray(bbox_image)
    }
    return ret


def make_markdown(ret, results_dir):
    image_path = f"{results_dir}/imgs/invig-test-{i:08d}.jpg"
    image = ret['image']
    image_size = np.asarray(image.size)
    image_size = image_size / image_size.min() * 300
    image.thumbnail(image_size)
    image.save(image_path)
    string = f"""
------
### images-test-{i:08d} ({'success' if ret['status'] else 'failure'})
![images-test-{i:08d}](imgs/invig-test-{i:08d}.jpg)
```
{ret['text']}
```

"""
    return string
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt", type=str)
    # parser.add_argument("--batch_size", default=32, type=int)
    main_args = parser.parse_args()

    # 1 load model
    path_ckpt = "/mnt/bn/ckpt-lq/vldd/invig_large_grounding_checkpoints/10_3e-5_512_20230420-0135/checkpoint_4_21000.pt"
    path_ckpt = main_args.ckpt
    model = OFAModelWarper(path_ckpt, args.path_ofa, args.path_tokenizer)
    logger.info(f"Test Oracle Acc on {path_ckpt}")

    # 2 load dataset
    path_dataset = "/mnt/bn/hri-lq/datasets/hf/guesswhat-success"
    split = "test"
    ds = TaskProcessor.setup_task(load_from_disk(path_dataset)[split], "guesswhat_grounding")
    ds = OFADataset(ds, config['path_images'], model.tokenizer, model.image_processor)
    # loader = torch.utils.data.DataLoader(ds, batch_size=main_args.batch_size, num_workers=1, collate_fn=ds.collater)

    # 3 benchmark
    results_dir = "./results"
    results_dir = f"{results_dir}/end2end-{time.strftime('%Y%m%d-%H%M')}"
    images_dir = f"{results_dir}/imgs"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    logger.info(f"results_dir: {results_dir}")

    records = []
    t = tqdm(range(len(ds)))
    for i in t:
        ret = make_a_dialog(ds[i], model)
        s = make_markdown(ret, results_dir)
        with open(f"{results_dir}/readme.md", "a") as f:
            f.write(s)
        records += [ret['status']]
        t.set_postfix({"SR": sum(records) / len(records)})
        # if i > 10:
        #     break

    with open(f"{results_dir}/readme.md", "a") as f:
        f.write(f"""
# SR = {sum(records) / len(records):.5f}
- ckpt - {path_ckpt}
- dataset - {path_dataset} ({split} split)
- size - {len(ds)}
""")
