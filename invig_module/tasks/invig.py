# Author: Jie Xu
# Date: 4/2/2023

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from tqdm import tqdm

import torch
from fairseq import metrics
from fairseq.tasks import register_task
from omegaconf import DictConfig

from tasks.ofa_task import OFATask, OFAConfig
from invig_module.data.dialog_dataset import OFADialogDataset
from invig_module.data.json_dataset import JsonlDataset

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
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "debug mode"}
    )


@register_task("invig", dataclass=InvigConfig)
class InvigTask(OFATask):
    def __init__(self, cfg: InvigConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.uses_ema = self.cfg.uses_ema

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))
        
        logger.info("source dictionary: {} types(invig)".format(len(src_dict)))
        logger.info("target dictionary: {} types(invig)".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        BASE_DIR = "/root/dialog_data"
        BASE_DIR = "/mnt/bn/hri-lq/datasets/ofa/dialog_data"
        # self.ds_invig     = JsonlDataset(f"{BASE_DIR}/invig_train.jsonl")
        # self.ds_guesswhat = JsonlDataset(f"{BASE_DIR}/guesswhat_train.jsonl")
        # self.ds_visdial   = JsonlDataset(f"{BASE_DIR}/visdial_train.jsonl")
        # self.ds_invig_val = JsonlDataset(f"{BASE_DIR}/invig_valid.jsonl")
        if split == 'train':
            # file_path = paths[(epoch - 1) % (len(paths) - 1)]
            logging.info(f"Loading {len(paths)-1} dataset(s): {paths[:-1]}")
            ds_lst = [JsonlDataset(file_path) for file_path in paths[:-1]]
            dataset = torch.utils.data.ConcatDataset(ds_lst)
            dataset.get_total_row_count = lambda: sum([d.get_total_row_count() for d in dataset.datasets])
            dataset._seek = lambda offset=0: [d._seek(offset) for d in dataset.datasets]
        else:
            # 载入测试集合
            file_path = paths[-1]
            dataset = JsonlDataset(file_path)

        # 给出几个数据集的例子
        # logger.info(f"Samples in dataset({split}): ")
        # for i in range(3):
        #     logger.info(f"{i}/{3}: {dataset[i]}")

        self.datasets[split] = OFADialogDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size
        )

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
    
    def build_model_loadbpe_only(self):
        bpe_dict = {
            "_name": "gpt2",
            "gpt2_encoder_json": os.path.join(self.cfg.bpe_dir, "encoder.json"),
            "gpt2_vocab_bpe": os.path.join(self.cfg.bpe_dir, "vocab.bpe")
        }
        bpe_dict = DictConfig(bpe_dict)
        self.bpe = self.build_bpe(bpe_dict)

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
        if self.uses_ema:
            assert 'ema_model' in extra_kwargs and extra_kwargs['ema_model'] is not None
        if self.uses_ema:
            eval_model = extra_kwargs['ema_model']
        else:
            eval_model = model

        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            hyps = hyps / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            refs = refs / (self.cfg.num_bins - 1) * self.cfg.max_image_size
            hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
            refs[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
            refs[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

            # scores = self._calculate_ap_score(hyps, refs)
            scores = self._calculate_ap_score(hyps, sample['region_coords'].float())
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
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(gen_out[i][0]["tokens"][:-1] - len(self.src_dict) + self.cfg.num_bins)
            refs.append(sample["target"][i][:-1] - len(self.src_dict) + self.cfg.num_bins)
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: ", hyps[0])
            logger.info("example reference: ", refs[0])

        return torch.stack(hyps, dim=0), torch.stack(refs, dim=0)
