# 基本设置
import argparse
import yaml

path_yaml = "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_env.yml"
with open(path_yaml) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
args = argparse.Namespace(**cfg['env'])

# 路径设置
import time
import copy
import os
import sys
import re
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
from PIL import Image
# import mmcv

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
from ofa_invig.evaluation import eval_grounding_acc_v2

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

MapFunc.load_config(path_yaml)


# =============== For Oracle ===============

def guesswhat_oracle(features, style=None):
    """ answer_invig """
    # 处理图片
    image = features['image']
    # 处理bbox
    bbox = features['bbox']
    sbbox = bbox_to_sbbox(bbox, *image.size)
    # 处理文本
    _dialog = features['dialog']
    _dialog = [j for i in _dialog for j in i]
    for m in range(1, len(_dialog) // 2):
        dialog = _dialog[:m * 2]

        dialogs = [["answer the question based on the region with yes or no."] + dialog]
        text_candidate = [MapFunc.make_text_pair(d, human_first=True, src_sbbox=sbbox) for d in dialogs]

        weights = [1]
        style = style if style else random.choices(range(len(text_candidate)), weights=weights)[0]
        src_text = text_candidate[style][0].lower()
        tgt_text = text_candidate[style][1].lower()
        # tgt_text = "not yet. " + tgt_text if style == 1 else tgt_text
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


# =============== For End2end ===============

from typing import Optional, Any, Union, List


def chat(model, images: List[Any], inputs: List[str]):
    collate_fn = DataCollatorForOFA(tokenizer=model.tokenizer, image_processor=model.image_processor)
    features = [{"src_text": txt, "tgt_text": "", "image": im} for im, txt in zip(images, inputs)]
    batch = collate_fn(features)
    text_gen = model.generate_origin(batch)
    return text_gen


def get_prompt(task, image, dialog, text, bbox):
    dialog = [j for i in dialog for j in i]
    assert bbox[2] > 1 and bbox[3] > 1
    if task == 'a':
        sbbox = bbox_to_sbbox(bbox, *image.size)
        if len(dialog) == 0:
            prompt = ["state your query about the region."] + [""]
        else:
            prompt = ["answer the question based on the region.", *dialog] + [""]
        prompt = MapFunc.make_text_pair(prompt, src_sbbox=sbbox)[0]
    elif task == 'q' or task == 'g':
        dialog += [text]
        prompt = ["can you specify which region the context describes?", *dialog] + [""]
        prompt = MapFunc.make_text_pair(prompt, human_first=True)[0]
    return prompt


from io import BytesIO
import requests
import base64

def chat_xvlm(task, image, dialog, text, bbox, all_bboxes, image_path=""):
    assert task in ['q', 'a', 'g']
    assert bbox[2] <= 1 and bbox[3] <= 1
    with BytesIO() as f:
        image.convert('RGB').save(f, format='jpeg')
        f.seek(0)
        image_b64 = base64.urlsafe_b64encode(f.read()).decode()

    data = {
        "token": "password",  # 简单校验用的token
        "task": task,
        "text": text,
        "dialog": dialog,
        "bboxes": json.dumps(all_bboxes),
        "ref_bboxes": json.dumps(np.asarray(bbox).reshape(4).tolist()),
        "image": image_b64,
        "image_path": image_path
    }
    api_url_map = {"q": "invig_test_questioner", "a": "invig_test_oracle", "g": "invig_test_guesser"}
    try:
        ret = requests.post(f"http://{main_args.ip}:5010/{api_url_map[task]}", json=data, timeout=10)
        ret = ret.json()['responce']
    except:
        logger.info(f"{ret.status_code}")
        print(ret.status_code)
        print(ret)
        raise
    return ret


if __name__ == "__main__":
    # python3 ofa_invig/bench_on_xvlm.py --task "invig_grounding_end2end" --comment "test" --oracle xvlm --agent xvlm --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230512-1515/checkpoint_last.pt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="invig_grounding", type=str)
    parser.add_argument("-c", "--ckpt", default="/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/ofa_large.pt", type=str)
    # parser.add_argument("--dataset", default="invig", type=str)
    parser.add_argument("--comment", default="", type=str)
    parser.add_argument("--log_dir", default="/mnt/bn/ckpt-lq/vldd/benchmark", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--oracle", default="xvlm", type=str)
    parser.add_argument("--agent", default="xvlm", type=str)
    parser.add_argument("--ip", default="", type=str)
    parser.add_argument("--start", default=0, type=int)
    main_args = parser.parse_args()
    print(main_args)

    # 0 prepare
    task_name = main_args.task
    path_ckpt = main_args.ckpt
    batch_size = main_args.batch_size
    comment = main_args.comment
    data_info = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(main_args.log_dir, task_name, comment)
    assert task_name in [
        'invig_grounding_end2end', 'invig_grounding'
    ]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)



    # 保存的内容：输入的句子、输出的句子、真正的句子、图片（路径？）(用序号代替)
    # 这里先实非end2end的
    MapFunc.load_config(path_yaml)
    
    try:
    # if 1:
        if task_name in ['invig_grounding']:
            batch_logs = [["index", "text_input", "text_output", "text_gen", "image_path", "bbox_target", "all_target"]]
            # 2 load dataset
            from ofa_invig.dialog_dataset import OFADataset
            test_config = MapFunc.cfg['test'][task_name]
            task = test_config['task']
            ds = datasets.load_from_disk(test_config['path'])['test']

            # 3 benchmark
            t = tqdm(ds)
            for i, d in enumerate(t):
                data = MapFunc.load_image(d)
                image_path = data['image_path']
                image = data['image']

                dialog = [j for i in data['dialog'] for j in i][1:]
                text_oracle = dialog[-1]
                dialog = [*zip(dialog[::2], dialog[1::2])]
                
                target = d['raw_target']
                all_bboxes = json.loads(d['raw_anns'])
                for ii, _ in enumerate(all_bboxes):
                    all_bboxes[ii]['bbox'] = (np.asarray(all_bboxes[ii]['bbox']) / np.asarray([*image.size, *image.size])).tolist()
                all_target = [all_bboxes[tt]['bbox'] for tt in target]
                bbox = all_target[0]
                sbbox = bbox_to_sbbox(bbox)

                # robot
                text_robot = chat_xvlm("g", image, dialog, text_oracle, bbox, all_bboxes, image_path)
                bbox_target = text_robot
                text_robot = bbox_to_sbbox(text_robot)
                batch_logs += [(i, dialog, sbbox, text_robot, image_path, bbox_target, all_target)]
            print(batch_logs[-1])

        elif task_name in ['invig_grounding_end2end']:

            # 1 load model
            model = OFAModelWarper(path_ckpt, cfg['env']['path_ofa'], path_yaml)
            logger.info(f"Test {task_name} on {path_ckpt}")
            MapFunc.load_config(path_yaml)

            batch_logs = [["index", "text_input", "text_output", "text_gen", "image_path", "bbox_target", "all_target"]]
            # 2 load dataset
            from ofa_invig.dialog_dataset import OFADataset
            test_config = MapFunc.cfg['test'][task_name]
            task = test_config['task']
            ds = datasets.load_from_disk(test_config['path'])['test']
            
            # 3 benchmark
            t = tqdm(ds)
            for i, d in enumerate(t):
                if i < main_args.start:
                    continue
                data = MapFunc.load_image(d)
                image_path = data['image_path']
                image = data['image']
                dialog = []

                all_target_ids = d['raw_target']
                all_bboxes = json.loads(d['raw_anns'])
                for ii, _ in enumerate(all_bboxes):
                    all_bboxes[ii]['bbox'] = (np.asarray(all_bboxes[ii]['bbox']) / np.asarray([*image.size, *image.size])).tolist()
                all_target = [all_bboxes[tt]['bbox'] for tt in all_target_ids]
                bbox = all_target[0]
                bbox_abs = bbox * np.asarray([*image.size, *image.size])
                sbbox = bbox_to_sbbox(bbox)

                for turn in range(8):  # max turn
                    # oracle
                    if main_args.oracle == "xvlm":
                        if len(dialog) == 0:
                            text_oracle = "i want that object."
                        else:
                            text_oracle = chat_xvlm("a", image, dialog, "", bbox, all_bboxes, image_path)
                    else:
                        prompt = get_prompt("a", image, dialog, "", bbox_abs)
                        text_oracle = chat(model, [image], [prompt])[0]

                    # robot/agent
                    if main_args.agent == "xvlm":
                        text_robot = chat_xvlm("q", image, dialog, text_oracle, bbox, all_bboxes, image_path)
                    else:
                        prompt = get_prompt("q", image, dialog, text_oracle, bbox_abs)
                        text_robot = chat(model, [image], [prompt])[0]
                        text_robot = re.sub("not yet\.", "", text_robot).strip()
                    dialog += [(text_oracle, text_robot)]
                        
                    # 终止条件
                    if main_args.agent == "xvlm" and text_robot == "":
                        break
                    if main_args.agent != "xvlm" and "region:" in text_robot:
                        break
                if main_args.agent == "xvlm":
                    bbox_target = chat_xvlm("g", image, dialog, text_oracle, bbox, all_bboxes, image_path)
                else:
                    bbox_target = sbbox_to_bbox(text_robot).tolist()
                batch_logs += [(i, dialog, sbbox, text_robot, image_path, bbox_target, all_target)]

        else:
            raise NotImplementedError
    except:
        logger.info(f"Except error!")

    # 4 save results
    batch_logs = [json.dumps(i)+"\n" for i in batch_logs]
    with open(f"{log_dir}/{data_info}.json", "w") as f:
        f.writelines(batch_logs)
    logger.info(f"Saved results to {log_dir}/{data_info}.json")
