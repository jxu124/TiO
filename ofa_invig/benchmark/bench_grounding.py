# Author: Xu Jie
# Date: 2023.03.25

import os
import dotenv

# load .env
ENV_FILE = os.path.join(os.path.dirname(__file__), "../.env")
assert dotenv.load_dotenv(ENV_FILE)

PATH_D_OFA = os.environ.get("PATH_D_OFA")
PATH_D_INVIG = os.environ.get("PATH_D_INVIG")
PATH_D_DATASET = os.environ.get("PATH_D_DATASET")

PATH_FAIRSEQ = f"{PATH_D_OFA}/fairseq"
PATH_TO_TF = "/mnt/bn/hri-lq/checkpoints/OFA-large"

import sys
sys.path.append(PATH_FAIRSEQ)
sys.path.append(PATH_D_OFA)

from PIL import Image
from tqdm import tqdm
import torch
import logging
import argparse
import numpy as np
import mmcv
import re

import fairseq
from fairseq import tasks, utils
from fairseq.tasks import FairseqTask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.utils import reset_logging

from transformers import OFATokenizer, OFAModel

utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/invig_module"}))

from invig_module.data.json_dataset import JsonlDataset
from invig_module.data.dialog_dataset import OFADialogDataset, sbbox_to_bbox
from utils import checkpoint_utils
from utils_evalution import eval_grounding_acc



logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

# argparse
parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-c', '--ckpt')
parser.add_argument('-b', '--batchsize', default=16)
args = parser.parse_args()


if __name__ == "__main__":
    # begin
    reset_logging()
    logger.info("Started.")
    # load user define module
    logger.info("#1 import_user_module")
    utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/invig_module"}))
    # load model, task, cfg
    logger.info("#2 load_model_ensemble_and_task")
    ckpt_path = args.ckpt
    # 1ds
    logger.info(f"ckpt file: {ckpt_path}")
    model_overrides = {"bpe_dir": f"{PATH_D_OFA}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    model = models[0]
    # load ema paramenters
    logger.info("#3 load_ema_from_checkpoint")
    model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    # get ready for model
    logger.info("#4 prepare_for_inference")
    model.eval()
    model.half()
    model.cuda()
    model.prepare_for_inference_(cfg)
    # done
    generator = task.build_generator(models, cfg.generation)
    logger.info("Done.")

    # tokenlizer (transformers)
    tokenizer = OFATokenizer.from_pretrained(PATH_TO_TF, local_files_only=True, truncation_side="left")

    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def restore_imgs(patch_images, w_resize_ratios, h_resize_ratios):
        mean = torch.Tensor([0.5, 0.5, 0.5])
        std = torch.Tensor([0.5, 0.5, 0.5])
        imgs = []
        for img, wr, hr in zip(patch_images, w_resize_ratios, h_resize_ratios):
            img = img.cpu()
            img = torch.permute(img, [1, 2, 0])
            img = (img * std + mean) * 255
            img = Image.fromarray(img.numpy().astype(np.uint8))
            w, h = img.size
            img = img.resize((w / wr, h / hr))
            imgs.append(img)
        return imgs

    prob_grounding = 1.0

    logger.info("Load dataset & make iterator.")
    dataset = JsonlDataset(f"{PATH_D_DATASET}/invig_test.jsonl")
    datasets = OFADialogDataset(
        "test",
        dataset,
        task.bpe,
        task.src_dict,
        task.tgt_dict,
        max_src_length=task.cfg.max_src_length,
        max_tgt_length=task.cfg.max_tgt_length,
        patch_image_size=task.cfg.patch_image_size,
        imagenet_default_mean_and_std=task.cfg.imagenet_default_mean_and_std,
        num_bins=task.cfg.num_bins,
        max_image_size=task.cfg.max_image_size
    )
    datasets.prob_grounding = prob_grounding
    itr = task.get_batch_iterator(
        dataset=datasets,
        max_tokens=cfg.dataset.max_tokens,
        # max_sentences=cfg.dataset.batch_size,
        max_sentences=int(args.batchsize),
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    grounding_imgs = []
    grounding_res = []
    grounding_ann = []
    grounding_src = []

    from collections import defaultdict
    counts = defaultdict(int)
    t = tqdm(itr)
    ii = 0
    for sample in t:
        ii += 1
        # if ii > 10:
        #     break
        # 从GT数据中恢复图片和bbox
        imgs = restore_imgs(sample["net_input"]["patch_images"], sample['w_resize_ratios'], sample['h_resize_ratios'])
        bboxes = sample['region_coords'].cpu().numpy().tolist()
        grounding_imgs += imgs
        grounding_ann += bboxes
        grounding_src += tokenizer.batch_decode(sample["net_input"]["src_tokens"], skip_special_tokens=True)
        
        # print( tokenizer.batch_decode(sample["net_input"]["src_tokens"], skip_special_tokens=True))
        # sample['prefix_tokens'] = torch.stack([tokenizer.encode("g: ", add_special_tokens=True, return_tensors="pt")[:-1]] * 16)
        # 模型预测
        sample = utils.move_to_cuda(sample)
        sample = utils.apply_to_sample(apply_half, sample)
        with torch.no_grad():
            hypos = task.inference_step(generator, models, sample, prefix_tokens=sample['prefix_tokens'])
        sbboxes = [tokenizer.decode(h[0]["tokens"]) for h in hypos]
        
        if ii == 1:
            print(sbboxes)
        bboxes_pred = []
        for img, sbbox in zip(imgs, sbboxes):
            counts["total"] += 1
            res = re.match(".*?(<bin_[0-9]*> <bin_[0-9]*> <bin_[0-9]*> <bin_[0-9]*>).*", sbbox)
            counts["no_pred"] += 1 if res is None else 0
            sbbox = res.group(1) if res is not None else "<bin_0> <bin_0> <bin_999> <bin_999>"
            bbox = sbbox_to_bbox(sbbox, img.size[0], img.size[1])
            bboxes_pred.append(bbox)
        grounding_res += bboxes_pred
        grounding_annotations = [{"ref_bboxes": np.array([bbox])} for bbox in grounding_ann]
        grounding_results = [{"pred_ref_bboxes": np.array([bbox])} for bbox in grounding_res]
        res = eval_grounding_acc(grounding_results, grounding_annotations)
        t.set_postfix({"ap_0.5": res['grounding_acc_iou_0.5']})
        # break
        
    # 计算acc
    print(counts)
    grounding_annotations = [{"ref_bboxes": np.array([bbox])} for bbox in grounding_ann]
    grounding_results = [{"pred_ref_bboxes": np.array([bbox])} for bbox in grounding_res]
    res = eval_grounding_acc(grounding_results, grounding_annotations)
    print(res)
