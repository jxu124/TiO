# Author: Xu Jie
# Date: 2023.03.25

import os
import dotenv

# load .env
ENV_FILE = os.path.join(os.path.dirname(__file__), "../../.env")
assert dotenv.load_dotenv(ENV_FILE)

PATH_D_OFA = os.environ.get("PATH_D_OFA")
PATH_D_INVIG = os.environ.get("PATH_D_INVIG")
PATH_D_DATASET = os.environ.get("PATH_D_DATASET")
PATH_D_LOG = os.environ.get("PATH_D_LOG")

PATH_FAIRSEQ = f"{PATH_D_OFA}/fairseq"
PATH_TO_TF = "/mnt/bn/hri-lq/checkpoints/OFA-large"

import sys
sys.path.append(PATH_FAIRSEQ)
sys.path.append(PATH_D_OFA)
sys.path.append(PATH_D_INVIG)

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

utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/ofa_invig/invig_module"}))
from invig_module.data.json_dataset import JsonlDataset
from invig_module.data.dialog_dataset import OFADialogDataset, sbbox_to_bbox
from invig_module.utils.evalution import eval_grounding_acc
from utils import checkpoint_utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



def load_ckpt(ckpt_path):
    """
    The load_ckpt function loads a model checkpoint from a file and returns a generator, model, configuration, and task object. The function first loads the user-defined module, then loads the model, task, and configuration from the checkpoint file. The function then loads the exponential moving average (EMA) parameters from the checkpoint file and prepares the model for inference. Finally, the function builds a generator object and returns the generator, model, configuration, and task objects.

    Here is a step-by-step explanation of how to use the load_ckpt function:

    Load the user-defined module. The load_user_module function loads the user-defined module, which is used to define the model architecture and the loss function.
    Load the model, task, and configuration from the checkpoint file. The load_model_ensemble_and_task function loads the model, task, and configuration from the checkpoint file. The model is a PyTorch model, the task is a definition of the problem to be solved, and the configuration is a set of parameters that control the training and inference process.
    Load the EMA parameters from the checkpoint file. The load_ema_from_checkpoint function loads the EMA parameters from the checkpoint file. The EMA parameters are used to smooth the gradients during training.
    Prepare the model for inference. The prepare_for_inference function prepares the model for inference. This includes setting the model to evaluation mode, converting the model to half-precision, and moving the model to the GPU.
    Build a generator object. The build_generator function builds a generator object. The generator object is used to generate samples from the model.
    Return the generator, model, configuration, and task objects. The load_ckpt function returns the generator, model, configuration, and task objects.

    """
    # load user define module
    logger.info("# 1.1 import_user_module")
    utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/ofa_invig/invig_module"}))
    # load model, task, cfg
    logger.info("# 1.2 load_model_ensemble_and_task")
    # 1ds
    logger.info(f"ckpt file: {ckpt_path}")
    model_overrides = {"bpe_dir": f"{PATH_D_OFA}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    print(task.inference_step)
    model = models[0]
    # load ema paramenters
    # logger.info("# 1.3 load_ema_from_checkpoint")
    # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    # prepare_for_inference
    logger.info("# 1.4 prepare_for_inference")
    model.eval().half().cuda()
    model.prepare_for_inference_(cfg)
    # build_generator
    logger.info("# 1.5 build_generator")
    generator = task.build_generator(models, cfg.generation)
    return generator, model, cfg, task


def get_dataloader(gens, batchsize=8, prob_grounding=1):
    generator, model, cfg, task = gens
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
        max_sentences=batchsize,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), model.max_positions()
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    return itr


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


def inference_step(gens, sample):  # tokenizer 
    generator, model, cfg, task = gens
    sample_hf = utils.apply_to_sample(apply_half, sample)
    sample_hf = utils.move_to_cuda(sample_hf)
    with torch.no_grad():
        hypos = task.inference_step(generator, [model], sample_hf)
    text_gen = tokenizer.batch_decode([h[0]['tokens'] for h in hypos], skip_special_tokens=True)
    return text_gen


def inference_dialog(gens, src_texts, patch_images):
    src_input = tokenizer.batch_encode_plus(src_texts, add_special_tokens=True, return_length=True, padding=True, return_tensors="pt")
    src_tokens = src_input["input_ids"]
    src_lengths = src_input["length"]
    patch_masks = torch.BoolTensor([True for _ in src_tokens])

    _sample = {
        "net_input": {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'patch_images': patch_images,
            'patch_masks': patch_masks,
        }
    }
    _text_gen = inference_step(gens, _sample)
    return _text_gen


def eval_gnding(gt, gen):
    grounding_annotations = [{"ref_bboxes": np.array([bbox])} for bbox in gt]
    grounding_results = [{"pred_ref_bboxes": np.array([bbox])} for bbox in gen]
    return eval_grounding_acc(grounding_results, grounding_annotations, with_gt_bbox=False)



