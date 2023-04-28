# Author: Jie Xu
# Date: 4/2/2023

from dataclasses import dataclass, field
import json
import logging
import os
import torch
import yaml
from typing import Optional
from argparse import Namespace
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import random

# fairseq的模块
from fairseq import metrics
from fairseq.tasks import register_task

# ofa的模块
from tasks.ofa_task import OFATask, OFAConfig

# invig的模块
from .dialog_dataset import MapFunc, DataCollatorForOFA, get_dataset
from .common import get_processor, world_info_from_env

# huggingface 的模块
import datasets


datasets.config.IN_MEMORY_MAX_SIZE = 1024000000
logger = logging.getLogger(__name__)


@dataclass
class InvigConfig(OFAConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "debug mode"}
    )
    config_yaml: Optional[str] = field(
        default="/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_env.yml",
        metadata={"help": "configure file"}
    )


class IterDatasetWrap(torch.utils.data.Dataset):
    def __init__(self, cfg, split="train"):
        self.ids = datasets.load_dataset(cfg['path'], streaming=True)[split]
        self.ids = datasets.distributed.split_dataset_by_node(self.ids, 0, 8)
        self.task = cfg['tasks'][0]
        self.n_samples = cfg['n_samples'] // 8
        self.it = None

    def __getitem__(self, idx):
        try:
            data = next(self.it)
        except:
            self.it = iter(self.ids)
            data = next(self.it)
        return getattr(MapFunc, self.task)(MapFunc.load_image(data))

    def set_epoch(self, epoch):
        self.ids.set_epoch(epoch)

    def __len__(self):
        return self.n_samples


@register_task("invig", dataclass=InvigConfig)
class InvigTask(OFATask):
    def __init__(self, cfg: InvigConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        MapFunc.load_config(self.cfg.config_yaml)
        self.processor = get_processor(pretrain_path=MapFunc.cfg['env']['path_tokenizer'], resolution=512)
        datasets.disable_progress_bar()

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # paths = self.cfg.data.split(',')
        # assert len(paths) > 0
        _split = "validation" if split == "valid" else split
        self.datasets[split] = get_dataset(_split)

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_acc:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )
        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            from .common import sbbox_to_bbox
            import numpy as np

            sbbox_gen, sbbox_gt = self._inference(self.sequence_generator, sample, model)
            bbox_gen = torch.Tensor(np.stack([sbbox_to_bbox(s).reshape(4) for s in sbbox_gen]))
            bbox_gt = torch.Tensor(np.stack([sbbox_to_bbox(s).reshape(4) for s in sbbox_gt]))
            scores = self._calculate_ap_score(bbox_gen, bbox_gt)
            logging_output["_score_sum"] = scores.sum().item()
            logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def _inference(self, generator, sample, model):
        gen_out = self.inference_step(generator, [model], sample)

        tokenizer = self.processor[0]
        sbbox_gen = tokenizer.batch_decode([h[0]['tokens'] for h in gen_out])
        sbbox_gt = tokenizer.batch_decode(sample['target'])
        if self.cfg.eval_print_samples:
            logger.info(f"example hyp(gen): {sbbox_gen[0]}")
            logger.info(f"example ref(gt):  {sbbox_gt[0]}")
        return sbbox_gen, sbbox_gt
