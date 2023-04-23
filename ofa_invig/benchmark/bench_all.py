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
PATH_D_LOG = os.environ.get("PATH_D_LOG")

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


def load_ckpt(ckpt_path):
    # load user define module
    logger.info("# 1.1 import_user_module")
    utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/invig_module"}))
    # load model, task, cfg
    logger.info("# 1.2 load_model_ensemble_and_task")
    # 1ds
    logger.info(f"ckpt file: {ckpt_path}")
    model_overrides = {"bpe_dir": f"{PATH_D_OFA}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    print(task.inference_step)
    model = models[0]
    # load ema paramenters
    logger.info("# 1.3 load_ema_from_checkpoint")
    model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    # prepare_for_inference
    logger.info("# 1.4 prepare_for_inference")
    model.eval()
    model.half()
    model.cuda()
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


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--ckpt')
    parser.add_argument('-b', '--batchsize', type=int, default=16)
    args = parser.parse_args()

    ckpt = args.ckpt
    batchsize = args.batchsize
    max_turn = 5

    # logging
    reset_logging()

    # 1 get generator, model, cfg, task
    logger.info("# 1 get model, cfg, task, generator")
    gens = load_ckpt(ckpt)  # generator, model, cfg, task 

    # 2 get tokenlizer (transformers)
    logger.info("# 2 get tokenlizer (transformers)")
    tokenizer = OFATokenizer.from_pretrained(PATH_TO_TF, local_files_only=True, truncation_side="left")

    # 3 get dataloader
    logger.info("# 3 get dataloader")
    loader = get_dataloader(gens, batchsize, prob_grounding=1.0)

    # 4 for sample:
    # 原始输入->输出grounding: 得到grounding-gt
    # 对话2-5轮，每轮结束后->输出grounding: 得到固定轮数grounding-turnx
    # 需要保存的东西：imgs, dialogs{"gt", "gen"}, bboxes{"gt", "gnding-gt", "gnding-t2", ..., "gnding-t5"} 结果 ap{"gnding-gt", "gnding-t2", ..., "gnding-t5"}
    # 循环1内：1 保存图片到imgs；2 保存gt到dialogs["gt"],bboxes["gt"]；3 生成gnding-gt保存到bboxes["gnding-gt"]；4 循环生成对话进入循环2
    # 循环2内：。。。
    logger.info("4 for sample:")

    # 初始化 imgs dialogs bboxes
    imgs = []
    dialogs = {"gt": [], "gen": []}
    bboxes = {"gt": [], "gnding-gt": [], "gnding-t2": [], "gnding-t3": [], "gnding-t4": [], "gnding-t5": []}

    t = tqdm(loader)
    for sample in t:
        # 1 保存图片到imgs
        _imgs = restore_imgs(sample["net_input"]["patch_images"], sample['w_resize_ratios'], sample['h_resize_ratios'])
        imgs += _imgs

        # 2 保存gt到dialogs["gt"], bboxes["gt"]
        # item in dialogs: ["", "<a0>", "<q1>", "<a1>", ...]
        _dialogs_gt = tokenizer.batch_decode(sample['net_input']['src_tokens'], skip_special_tokens=True)
        dialogs["gt"] += [re.split("Output the region described. | q: | a: ", d) for d in _dialogs_gt]
        _sbboxes_gt = tokenizer.batch_decode(sample['target'], skip_special_tokens=True)  # 方式1：从target中反向获取
        _bboxes_gt = [sbbox_to_bbox(s, _imgs[i].size[0], _imgs[i].size[1]).tolist() for i, s in enumerate(_sbboxes_gt)]
        # _bboxes_gt = sample['region_coords'].cpu().numpy().tolist()  # 方式2：直接从sample中的region_coords拿
        bboxes["gt"] += _bboxes_gt
        
        # 3 生成gnding-gt保存到bboxes["gnding-gt"]
        _sbboxes_gen_0 = inference_step(gens, sample)
        _bboxes_gen_0 = [sbbox_to_bbox(s, _imgs[i].size[0], _imgs[i].size[1]).tolist() for i, s in enumerate(_sbboxes_gen_0)]
        bboxes["gnding-gt"] += _bboxes_gen_0
        
        # 4 循环生成对话进入循环2
        # item in dialogs: ["", "<a0>", "<q1>", "<a1>", ...]
        dialogs_gen = [[""] for _ in sample['target']]
        patch_images = sample["net_input"]["patch_images"]
        for turn in range(1, max_turn+1):
            # infer dialog
            if turn == 1:
                # a: descript
                src_texts = [f'Output an answer. Given {re.sub("g: ", "", _s)} . q: what do you see?' for _s in _sbboxes_gt]
                gen_texts = [re.sub("a: ", "", d) for d in inference_dialog(gens, src_texts, patch_images)]  # 第一轮对话没有“a:”前缀
                [d.append(gen_texts[i]) for i, d in enumerate(dialogs_gen)]
                continue
            else:
                # q
                src_texts = [f'Output a question.{" ".join(d)}' for d in dialogs_gen]
                gen_texts = inference_dialog(gens, src_texts, patch_images)
                [d.append(gen_texts[i]) for i, d in enumerate(dialogs_gen)]
                # a
                src_texts = [f'Output an answer. Given {re.sub("g: ", "", _s)} .{" ".join(d)}' for _s, d in zip(_sbboxes_gt, dialogs_gen)]
                gen_texts = inference_dialog(gens, src_texts, patch_images)
                [d.append(gen_texts[i]) for i, d in enumerate(dialogs_gen)]
            # infer grounding
            src_texts = [f'Output the region described.{" ".join(d)}' for d in dialogs_gen]
            _sbboxes_gen = inference_dialog(gens, src_texts, patch_images)
            _bboxes_gen = [sbbox_to_bbox(s, _imgs[i].size[0], _imgs[i].size[1]).tolist() for i, s in enumerate(_sbboxes_gen)]
            bboxes[f"gnding-t{turn}"] += _bboxes_gen
        dialogs["gen"] = dialogs_gen
        # 实时显示，让人更耐心一点！
        t.set_postfix({
            "gt": eval_gnding(bboxes["gt"], bboxes["gnding-gt"])['grounding_acc_iou_0.5'],
            "t2": eval_gnding(bboxes["gt"], bboxes["gnding-t2"])['grounding_acc_iou_0.5'],
            "t3": eval_gnding(bboxes["gt"], bboxes["gnding-t3"])['grounding_acc_iou_0.5'],
            "t4": eval_gnding(bboxes["gt"], bboxes["gnding-t4"])['grounding_acc_iou_0.5'],
            "t5": eval_gnding(bboxes["gt"], bboxes["gnding-t5"])['grounding_acc_iou_0.5'],
        })
        # break
        
    gnding_acc = {}
    logger.info("*"*40)
    logger.info("  Result")
    logger.info("*"*40)
    gnding_acc['gt'] = eval_gnding(bboxes["gt"], bboxes["gnding-gt"])
    logger.info(f'gt: {gnding_acc["gt"]}')
    for i in range(2, max_turn+1):
        gnding_acc[f't{i}'] = eval_gnding(bboxes["gt"], bboxes[f"gnding-t{i}"])
        logger.info(f't{i}: {gnding_acc[f"t{i}"]}')

    import pickle, time
    time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    filepath = os.path.join(PATH_D_LOG, f"result-{time_str}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump({"imgs": imgs, "dialogs": dialogs, "bboxes": bboxes, "gnding_acc": gnding_acc, "ckpt": ckpt}, f)
    logger.info(f'Save results in {filepath} ! Rename it!')

    # launch --type a100-80g --memory 160 -- python3 OFA-Invig/test/bench_all.py -b 64 -c OFA-checkpoints/bk-230402/invig_large_checkpoints/20_3e-5_512_20230325-1127/checkpoint_15_58000.pt
    # launch --type a100-80g --memory 160 -- python3 OFA-Invig/test/bench_all.py -b 16 -c OFA-checkpoints/bk-230402/invig_huge_checkpoints/10_3e-5_512_20230326-0235/checkpoint_8_58000.pt
    # large 16 : 20GB
