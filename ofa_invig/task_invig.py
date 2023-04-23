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

# fairseq的模块
from fairseq import metrics
from fairseq.tasks import register_task

# ofa的模块
from tasks.ofa_task import OFATask, OFAConfig

# invig的模块
from .dialog_dataset import MapFunc, DataCollatorForOFA, GenSampler
from .common import get_processor, world_info_from_env

# huggingface 的模块
from datasets.distributed import split_dataset_by_node
from datasets import load_dataset, load_from_disk
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


# hack fairseq
from datasets import load_from_disk, Features, Value
from fairseq.data import FairseqDataset
        
        
# iterable 数据集：包装为普通数据集

import bisect
import argparse
class OFADataset(torch.utils.data.ConcatDataset, FairseqDataset):  # ~OFADataset
    def __init__(self, datasets, tasks=None, tokenizer=None, image_processor=None):
        _, rank, world_size = world_info_from_env()
        total_count = sum([len(ds) for ds in datasets])
        self.ds_list = [split_dataset_by_node(ds, rank, world_size) for ds in datasets]
        self.dataset = argparse.Namespace(**{'get_total_row_count': lambda: total_count, "_seek": lambda offset=0: None})
        super().__init__(self.ds_list)
        self.tasks = tasks
        self.tokenizer, self.image_processor = tokenizer, image_processor

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data = self.datasets[dataset_idx][sample_idx]
        if self.tasks is not None:
            return getattr(MapFunc, self.tasks[dataset_idx])(MapFunc.load_image(data))
        else:
            return MapFunc.load_image(data)
    
    def __len__(self):
        return super().__len__()

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.ds_list:
            if hasattr(ds, "shuffle"):
                ds.shuffle(epoch)
            elif hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def collater(self, samples, pad_to_length=None):
        collate_fn = DataCollatorForOFA(tokenizer=self.tokenizer, image_processor=self.image_processor)
        return collate_fn(samples)


@register_task("invig", dataclass=InvigConfig)
class InvigTask(OFATask):
    def __init__(self, cfg: InvigConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.processor = get_processor(resolution=512)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # paths = self.cfg.data.split(',')
        # assert len(paths) > 0
        _split = "validation" if split == "valid" else split

        # 1 按照node切分数据集
        datasets_collection = []
        for i in MapFunc.cfg[_split]:
            if i.get('streaming'):
                continue
            name = i['name']
            logger.info(f"loading {name}-{_split} from {i['path']}")
            ds = datasets.load_from_disk(i['path'])[_split]
            ds = ds.filter(MapFunc.filter_exclude_test_images, input_columns=['global_image_id'])
            for task, prob in zip(i['tasks'], i['probs']):
                datasets_collection += [(name, task, ds, prob)]

        # 2 按照比例组合数据
        use_datasets = [i[2] for i in datasets_collection]
        probs = np.asarray([i[3] for i in datasets_collection])
        probs = probs / probs.sum()
        counts = np.asarray([len(ds) for ds in use_datasets])
        total_count = np.min(counts / probs) if split == 'train' else 1024
        n_samples = (total_count * probs).astype(int)
        use_datasets = [ds.shuffle(keep_in_memory=True).select(range(n)) for ds, n in zip(use_datasets, n_samples)]
        tasks = [i[1] for i in datasets_collection]

        # 3 构造fairseq_dataset，送进去预处理函数
        dataset = OFADataset(use_datasets, tasks, *self.processor)
        self.datasets[split] = dataset

        # 4 打印logs
        logger.info(f"load {split} data: {len(use_datasets)} ({len(dataset)} samples) dataset(s)")
        logger.info("Tasks: " + ", ".join([f'{t}({n})' for t, n in zip(tasks, n_samples)]))

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
