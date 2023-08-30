# 基本设置
import argparse

# 路径设置
import os
import sys
sys.path.append("./attachments/OFA")

# 载入模块
from tqdm.auto import tqdm
import logging
import numpy as np
import torch

# 载入OFA的模块 自己invig的模块
from tio_core.utils.training import world_info_from_env
from tio_core.module.model import OFAModelWarper
from tio_core.utils.data import get_processor

# 载入transformers
import transformers
import datasets


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


if __name__ == "__main__":
    DEBUG_MODE = True
    # DEBUG_MODE = False
    local_rank, rank, world_size = world_info_from_env()

    # load model
    model = OFAModelWarper()

    # load dataset
    from tio_core.utils.config import TiOConfig
    from tio_core.module.dataset import TiODataset
    datasets.disable_progress_bar()
    config = TiOConfig("/mnt/bn/hri-lq/projects/TiO/config/training.yml")
    tokenizer, image_processor = config.get_processor()
    ds_train = config.get_dataset("train")
    ds_train = TiODataset(ds_train, config)

    hyper_args = argparse.Namespace(**{
        "per_device_train_batch_size": 3,
        # "per_device_eval_batch_size": 6,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "num_train_epochs": 10,
        "warmup_steps": 3000,
        # "label_smoothing_factor": 0.1,
    })
    setting_args = argparse.Namespace(**{
        "remove_unused_columns": False,
        "fp16": True,
        "fp16_full_eval": True,
        "dataloader_num_workers": 8,
        "logging_steps": 20 if DEBUG_MODE else 50,
        "report_to": "none" if DEBUG_MODE else "all",
        "output_dir": "./runs-debug" if DEBUG_MODE else "./runs",
        # "label_names": ["target"],
        "evaluation_strategy": "steps",
        "eval_steps": 20 if DEBUG_MODE else 200,
        "save_strategy": "steps",
        "save_steps": 40 if DEBUG_MODE else 200,
        "save_total_limit": 20,
        "fsdp": "shard_grad_op",
        # "load_best_model_at_end": True,
        # "ddp_find_unused_parameters": True,
    })
    if DEBUG_MODE:
        setting_args.__delattr__("fsdp")

    from tio_core.utils.data_collator import DataCollator
    trainer = transformers.Trainer(
        model=model,
        train_dataset=ds_train,
        # eval_dataset=ds_valid,
        data_collator=ds_train.collater,
        args=transformers.TrainingArguments(**vars(hyper_args), **vars(setting_args)),
        # callbacks=[MyCallback()]
    )
    trainer.train()
