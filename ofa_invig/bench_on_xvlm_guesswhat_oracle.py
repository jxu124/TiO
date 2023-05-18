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
import random
from PIL import Image

# 载入transformers
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

# 载入OFA的模块 自己invig的模块
# from utils import checkpoint_utils
# from ofa_invig.dialog_dataset import MapFunc

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


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
    ret = requests.post(f"http://[fdbd:dc03:9:33:1200::46]:5010/{api_url_map[task]}", json=data, timeout=10)
    ret = ret.json()['responce']
    return ret


if __name__ == "__main__":
    # python3 ofa_invig/bench_on_xvlm.py --task "invig_grounding_end2end" --comment "test" --oracle xvlm --agent xvlm --ckpt "/mnt/bn/ckpt-lq/vldd/invig_ablation_checkpoints/10_3e-5_512_20230512-1515/checkpoint_last.pt"
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="invig", type=str)
    parser.add_argument("--comment", default="", type=str)
    parser.add_argument("--log_dir", default="/mnt/bn/ckpt-lq/vldd/benchmark", type=str)
    main_args = parser.parse_args()

    # 0 prepare
    task_name = "xvlm_gw_oracle"
    comment = main_args.comment
    data_info = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(main_args.log_dir, task_name, comment)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{log_dir}/data_info"


    # 保存的内容：输入的句子、输出的句子、真正的句子、图片（路径？）(用序号代替)
    # 这里先实非end2end的


    batch_logs = [["index", "text_input", "text_output", "text_gen", "image_path", "bbox_target", "all_target"]]
    # 2 load dataset
    ds = datasets.load_from_disk("/mnt/bn/hri-lq/datasets/hf-cache/guesswhat")['test']

    # 3 benchmark
    t = tqdm(ds)
    for i, d in enumerate(t):
        image_path = d['image_path']
        _image_path = image_path.replace("coco", "/mnt/bn/hri-lq/datasets/images/coco")
        image = Image.open(_image_path)
        dialog = [""] + [j for i in d['dialog'] for j in i] + [""]
        dialog = [*zip(dialog[::2], dialog[1::2])]

        all_bboxes = json.loads(d['raw_objects'])
        for ii, _ in enumerate(all_bboxes):
            all_bboxes[ii]['bbox'] = (np.asarray(all_bboxes[ii]['bbox']) / np.asarray([*image.size, *image.size])).tolist()
        bbox = (np.asarray(d['bbox']) / np.asarray([*image.size, *image.size])).tolist()
        print(bbox)
        print(type(bbox))
        print(all_bboxes)

        for i in range(1, len(dialog)):
            _dialog = dialog[:i]
            text_oracle = dialog[i][0]
            text_robot = chat_xvlm("g", image, _dialog, text_oracle, bbox, all_bboxes, image_path)
            print(_dialog)
            print(text_oracle)
            print(text_robot)
            print()
        
        break
        

        # # robot
        # text_robot = chat_xvlm("g", image, dialog, text_oracle, bbox, all_bboxes, image_path)
        # text_robot = bbox_to_sbbox(text_robot)
        # bbox_target = text_robot
        # batch_logs += [(i, dialog, sbbox, text_robot, image_path, bbox_target, all_target)]
    

    # 4 save results
    # batch_logs = [json.dumps(i)+"\n" for i in batch_logs]
    # with open(f"{log_dir}/{data_info}.json", "w") as f:
    #     f.writelines(batch_logs)
    # logger.info(f"Saved results to {log_dir}/{data_info}.json")
