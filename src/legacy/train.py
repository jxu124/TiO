import os
import sys
import dotenv
assert dotenv.load_dotenv("/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/.env")

PATH_D_OFA = os.environ['PATH_D_OFA']
PATH_D_INVIG = os.environ['PATH_D_INVIG']
sys.path.append(PATH_D_OFA)
os.chdir(PATH_D_INVIG)

# 载入模块
import argparse
import logging
import numpy as np
from tqdm.auto import tqdm
# from torch.cuda.amp import autocast as autocast
import torch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{PATH_D_INVIG}/src"}))

# 载入transformers
from transformers import OFATokenizer, OFAModel
import datasets

# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.common import sbbox_to_bbox, get_processor
from src.legacy.dialog_dataset import collate_ofa_fairseq
from src.evaluation import eval_grounding_acc_v2


def load_ckpt(ckpt_path, dtype=torch.float32, device="cpu"):
    logger.info("开始载入model参数...")
    model_overrides = {"bpe_dir": f"{PATH_D_OFA}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    model = models[0]
    logger.info("开始载入ema参数...")
    # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    logger.info("模型设置中...")
    model.eval().to(dtype).to(device)
    # model.prepare_for_inference_(cfg)
    generator = task.build_generator(models, cfg.generation)
    logger.info("完成！")
    return generator, model, cfg, task

# load model
ckpt_path = "/mnt/bn/hri-lq/projects/VLDD/OFA-checkpoints/bk-230402/invig_large_checkpoints/20_3e-5_512_20230325-1127/checkpoint_15_56000.pt"
ckpt_path = "/mnt/bn/ckpt-lq/vldd/invig_large-3ds_checkpoints/30_1e-4_512_20230404-1407/checkpoint_20_75000.pt"
ckpt_path = "/mnt/bn/hri-lq/projects/VLDD/OFA-checkpoints/bk-230402/invig_2ds_checkpoints/20_3e-5_512_20230325-1126/checkpoint_10_19000.pt"
gens = load_ckpt(ckpt_path, "cpu")
nerator, model, cfg, task = gens
# load processor
tokenizer, image_processor = get_processor(resolution=512)

from torch.utils.data import DataLoader
from datasets import load_from_disk
from src.legacy.dialog_dataset import TaskProcessor, OFADataset

tokenizer, img_processor = get_processor(resolution=512)

ds_guesswhat = load_from_disk(f"/mnt/bn/hri-lq/datasets/hf/guesswhat")
ds_train = TaskProcessor.setup_task(ds_guesswhat['train'], "grounding")
ds_train = OFADataset(ds_train, tokenizer, img_processor)

ds_valid = TaskProcessor.setup_task(ds_guesswhat['validation'], "grounding")
ds_valid = OFADataset(ds_valid, tokenizer, img_processor)

from torch.utils.data import DataLoader
from transformers import get_scheduler
import lightning.pytorch as pl

class LitOFA(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        apply_fp16 = lambda t: t.to(dtype=torch.float16) if t.dtype is torch.float32 else t
        sample_hf = utils.apply_to_sample(apply_fp16, batch)
        sample_hf = utils.move_to_cuda(sample_hf)

        net_output = model(**sample_hf['net_input'])
    
        lprobs = model.get_normalized_probs(net_output, log_probs=True).permute([0, 2, 1])
        target = model.get_targets(sample_hf, net_output)
        loss = torch.nn.functional.cross_entropy(lprobs, target, ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=0.1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
# loader
train_dataloader = DataLoader(ds_train, shuffle=True, batch_size=3, collate_fn=ds_train.collater, num_workers=4)
# eval_dataloader = DataLoader(ds_valid, batch_size=2, collate_fn=ds_valid.collater, num_workers=4)

# model
lit_ofa = LitOFA(model)

# trainer
trainer = pl.Trainer(
    accelerator="auto",
    precision='16',
    # devices=1,
    # max_steps=20,
    accumulate_grad_batches=4,
    # enable_checkpointing=False,
    max_epochs=5
)

# train
trainer.fit(model=lit_ofa, train_dataloaders=train_dataloader)
