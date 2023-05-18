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
sys.path.append(cfg['env']['path_ofa'])
os.chdir(cfg['env']['path_invig'])

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
import logging

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{cfg['env']['path_invig']}/ofa_invig"}))

# 载入OFA的模块 自己invig的模块
from ofa_invig.dialog_dataset import MapFunc, DataCollatorForOFA
from ofa_invig.common import OFAModelWarper, bbox_to_sbbox, sbbox_to_bbox
# from ofa_invig.

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

MapFunc.load_config(path_yaml)


class VDG():
    def __init__(self, path_ckpt):
        # 1 load model
        if type(path_ckpt) is str:
            self.model = OFAModelWarper(path_ckpt, cfg['env']['path_ofa'], path_yaml)
        else:
            self.model = path_ckpt
        self.collate_fn = DataCollatorForOFA(tokenizer=self.model.tokenizer, image_processor=self.model.image_processor)

    def chat(self, 
             images: List[Any], 
             inputs: List[str]):
        features = [{"src_text": txt, "tgt_text": "", "image": im} for im, txt in zip(images, inputs)]
        batch = self.collate_fn(features)
        # text_gt = self.model.tokenizer.batch_decode(batch['target'], skip_special_tokens=True)
        text_gen = self.model.generate_origin(batch)
        return text_gen
