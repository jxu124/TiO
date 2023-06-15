# Author: ***
# Date: 4/2/2023
# This file is modified from https://github.com/OFA-Sys/OFA/blob/main/tasks/mm_tasks/refcoco.py

from dataclasses import dataclass, field
import json
import logging
import torch
from typing import Optional
from argparse import Namespace
from omegaconf import DictConfig
import numpy as np
import datasets

# ***
from fairseq import metrics
from fairseq.tasks import register_task

# ***
from tasks.ofa_task import OFATask, OFAConfig

# ***
from tio_utils import TiOConfig, sbbox_to_bbox
from .dataset import TiODataset


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
        default="",
        metadata={"help": "configure file"}
    )


@register_task("invig", dataclass=InvigConfig)
class InvigTask(OFATask):
    _tio_config = None

    def __init__(self, cfg: InvigConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    @property
    def tio_config(self):
        if self._tio_config is None:
            logger.info(f"load config_yaml: {self.cfg.config_yaml}")
            self._tio_config = TiOConfig(self.cfg.config_yaml)
        return self._tio_config

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # assert len(paths) > 0
        datasets.disable_progress_bar()
        dataset = self.tio_config.get_dataset("validation" if split == "valid" else split)
        dataset = TiODataset(dataset, self.tio_config, distributed_split=True)
        self.datasets[split] = dataset

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
        tokenizer = self.tio_config.tokenizer
        sbbox_gen = tokenizer.batch_decode([h[0]['tokens'] for h in gen_out])
        sbbox_gt = tokenizer.batch_decode(sample['target'])
        if self.cfg.eval_print_samples:
            logger.info(f"example hyp(gen): {sbbox_gen[0]}")
            logger.info(f"example ref(gt):  {sbbox_gt[0]}")
        return sbbox_gen, sbbox_gt
