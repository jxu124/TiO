# Author: Jie Xu
# Date: 4/2/2023

from dataclasses import dataclass, field
import json
import logging
import os
import torch
from typing import Optional
from argparse import Namespace
from tqdm import tqdm
from omegaconf import DictConfig

# fairseq的模块
from fairseq import metrics
from fairseq.tasks import register_task

# ofa的模块
from tasks.ofa_task import OFATask, OFAConfig

# invig的模块
from .dialog_dataset import OFADataset, TaskProcessor
from .common import get_processor

# huggingface 的模块
from datasets import load_dataset, concatenate_datasets, interleave_datasets, load_from_disk
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


@register_task("invig", dataclass=InvigConfig)
class InvigTask(OFATask):
    def __init__(self, cfg: InvigConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.processor = get_processor(resolution=512)
        self.hf_dataset = {}
        import yaml
        with open('/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/config/invig_4ds.yml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0
        _map = {"valid": "validation"}
        _split = _map.get(split, split)

        use_datasets, probs = [], []
        for i in self.config[_split]:
            name = i['name']
            if name not in self.hf_dataset:
                self.hf_dataset[name] = load_from_disk(i['path'])
            use_datasets += [TaskProcessor.setup_task(self.hf_dataset[name][_split], task) for task in i['tasks']]
            probs += i['probs']

        if _split == "train":
            # dataset = interleave_datasets(use_datasets, probs, seed=42, stopping_strategy="all_exhausted")
            dataset = concatenate_datasets(use_datasets).shuffle(seed=42+epoch)
        else:
            dataset = concatenate_datasets(use_datasets).shuffle(seed=42+epoch).select(range(1024))
            
        logger.info(f"load {split} data: {len(use_datasets)} dataset(s) within {len(dataset)} samples ")
        logger.info(f"load {split} data: {self.config[_split]}")

        # hack filedataset
        total_row_count = len(dataset)
        self.datasets[split] = OFADataset(dataset, self.config['path_images'], *self.processor)
        self.datasets[split].dataset.get_total_row_count = lambda: total_row_count
        self.datasets[split].dataset._seek = lambda offset=0: None

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
            bbox_gen = torch.Tensor(np.stack([sbbox_to_bbox(s) for s in sbbox_gen]))
            bbox_gt = torch.Tensor(np.stack([sbbox_to_bbox(s) for s in sbbox_gt]))
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
