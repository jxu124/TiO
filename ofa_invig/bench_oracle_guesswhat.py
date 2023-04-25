# 基本设置
import argparse
import yaml

path_yaml = "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_env.yml"
with open(path_yaml) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
args = argparse.Namespace(**cfg['env'])

# 路径设置
import os
import sys
import json
sys.path.append(cfg['env']['path_ofa'])
os.chdir(cfg['env']['path_invig'])

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
import logging
import numpy as np
import torch
import random

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{cfg['env']['path_invig']}/ofa_invig"}))

# 载入transformers
from ofa_invig.ofa.tokenization_ofa import OFATokenizer
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

        
# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from ofa_invig.dialog_dataset import MapFunc, DataCollatorForOFA
from ofa_invig.common import world_info_from_env, get_processor, sbbox_to_bbox, bbox_to_sbbox
from ofa_invig.common import OFAModelWarper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

MapFunc.load_config(path_yaml)


def guesswhat_oracle(features, style=None):
    """ answer_invig """
    # 处理图片
    image = features['image']
    # 处理bbox
    bbox = features['bbox']
    sbbox = bbox_to_sbbox(bbox, *image.size)
    # 处理文本
    dialog = json.loads(features['dialog'])
    for turn in range(1, len(dialog)+1):
        context = ["query: guess what i want."]
        for t in range(turn-1):
            context += [f"question: {dialog[t][0]}"]  # question
            context += [f"answer: {dialog[t][1]}"]  # answer
        # context = ["agent: guess what i want."]
        # for t in range(turn-1):
        #     context += [f"human: {dialog[t][0]}"]  # question
        #     context += [f"agent: {dialog[t][1]}"]  # answer
        context = " ".join(context)
        question = dialog[turn-1][0]
        answer = dialog[turn-1][1]
        based_on_the_region = "based on the region " if random.random() > 0.5 else ""

        src_candidate = [
            f" \n#instruction: answer the question {based_on_the_region}with yes or no. {question}\n#region: {sbbox}\n#context: \"{context}\"",
            f" \n#instruction: {question} yes or no?\n#region: {sbbox}\n#context: \"{context}\"",
        ]
        tgt_candidate = [
            f" {answer}",
            f" {answer}",
        ]
        weights = [7, 3]
        style = style if style else random.choices(range(len(src_candidate)), weights=weights)[0]
        src_text = src_candidate[style].lower()
        tgt_text = tgt_candidate[style].lower()
        assert image and question and answer
        yield {"src_text": src_text, "tgt_text": tgt_text, "image": image}


class OracleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.dataset = dataset
        else:  # in a worker process
            self.dataset = datasets.distributed.split_dataset_by_node(self.dataset, rank=worker_info.id, world_size=worker_info.num_workers)

    def __iter__(self):
        for data in self.dataset:
            yield from guesswhat_oracle(MapFunc.load_image(data), style=0)


if __name__ == "__main__":
    # large
    # A100-80g: batch_size=128
    # V100-32g: batch_size=16
    # huge
    # A100-80g: batch_size=128
    # V100-32g: batch_size=16
    # python3 ofa_invig/bench_oracle_guesswhat.py -c /mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230425-0955/checkpoint_1_6000.pt --batch_size 16
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt", type=str)
    parser.add_argument("--batch_size", default=48, type=int)
    # main_args = parser.parse_args(["-c", "/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt"])
    # main_args = parser.parse_args(["-c", "/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230425-0955/checkpoint_1_5000.pt"])
    main_args = parser.parse_args()

    # 1 load model
    path_ckpt = main_args.ckpt
    model = OFAModelWarper(path_ckpt, cfg['env']['path_ofa'], path_yaml)
    logger.info(f"Test OracleGuesswhat on {path_ckpt}")

    # 2 load dataset
    ds = datasets.load_from_disk("/mnt/bn/hri-lq/datasets/hf-cache/guesswhat")['test']
    ds = ds.add_column("__task__", ["guesswhat_oracle"]*len(ds))  # .filter(MapFunc.filter_exclude_test_images, input_columns=['global_image_id'])
    ids = OracleDataset(ds)
    collate_fn = DataCollatorForOFA(tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader = torch.utils.data.DataLoader(ids, batch_size=main_args.batch_size, num_workers=8, collate_fn=collate_fn)

    # 3 benchmark
    t = tqdm(loader)
    texts_gt = []
    texts_gen = []
    history = []
    for i, b in enumerate(t):
        text_gt = model.tokenizer.batch_decode(b['target'], skip_special_tokens=True)
        text_gen = model.generate_origin(b)
        texts_gt += text_gt
        texts_gen += text_gen
        history += [a[1:2] == b[1:2] for a, b in zip(text_gen, text_gt)]
        if i % 10 == 0:
            acc = np.sum(history) / len(history)
            t.set_postfix({"acc": acc})
        # if i > 10:
        #     break
    acc = np.sum(history) / len(history)
    logger.info(f"OracleAcc: {acc:.3f}")

    # 4 save results
    with open("./eval_bench_oracle_guesswhat.log", 'w') as f:
        json.dump({"acc": acc, "texts_gt": texts_gt, "texts_gen": texts_gen, "history": history}, f)
    logger.info(f"Saved results to ./eval_oracle.log")
