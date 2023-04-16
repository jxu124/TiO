# 基本设置
import argparse
args = argparse.Namespace(**{
    "path_ofa": "/mnt/bn/hri-lq/projects/VLDD/OFA",
    "path_invig": "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig",
})

# 路径设置
import os
import sys
sys.path.append(args.path_ofa)
sys.path.append(args.path_ofa + "/fairseq")
os.chdir(args.path_invig)

# 载入模块
from tqdm.auto import tqdm
from typing import Optional, Any, Union, List
from dataclasses import dataclass
from torch.cuda.amp import autocast
import logging
import numpy as np
import torch

# 载入fairseq
from fairseq import utils
utils.import_user_module(argparse.Namespace(**{"user_dir": f"{args.path_invig}/src"}))

# 载入transformers
from src.ofa.tokenization_ofa import OFATokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk
import transformers
import datasets

# 载入OFA的模块 自己invig的模块
from utils import checkpoint_utils
from src.dialog_dataset import TaskProcessor
from src.common import world_info_from_env, get_image_processor, sbbox_to_bbox
# from src.common import sbbox_to_bbox, get_image_processor
# from src.dialog_dataset import collate_ofa_fairseq
from src.evaluation import eval_grounding_acc_v2


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

def load_from_pretrain(ckpt_path):
    logger.info("开始载入model参数...")
    model_overrides = {"bpe_dir": f"{args.path_ofa}/utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
    model = models[0]
    logger.info("开始载入ema参数...")
    # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
    logger.info("模型设置中...")
    model.eval()
    # model.prepare_for_inference_(cfg)
    generator = task.build_generator(models, cfg.generation)
    logger.info("完成！")
    return generator, model, cfg, task


# model warper
class OFAModelWarper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, **batch):
        net_output = self.model(**batch['net_input'])
        lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
        target = self.model.get_targets(batch, net_output)
        loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1), ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=0.1)
        return Seq2SeqLMOutput(loss=loss, logits=lprobs)
        

@dataclass
class DataCollatorForOFA(transformers.DataCollatorForSeq2Seq):
    tokenizer: transformers.PreTrainedTokenizerBase
    image_processor: Optional[Any] = None
    padding: Optional[str] = "longest"
    max_src_length: Optional[int] = 256
    max_tgt_length: Optional[int] = 64
    return_tensors: Optional[str] = "pt"
    
    def __call__(self, features, return_tensors=None):
        src_texts = [f[0] for f in features]
        tgt_texts = [f[1] for f in features]
        patch_images = torch.stack([self.image_processor(f[2]) for f in features])
        
        # max_length longest
        inputs = self.tokenizer(text=src_texts, return_length=True, max_length=self.max_src_length, padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        labels = self.tokenizer(text_target=tgt_texts, max_length=self.max_tgt_length, padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        patch_masks = torch.tensor([True] * len(patch_images))
        prev_output_tokens = labels.input_ids[..., :-1]
        target = labels.input_ids[..., 1:]
        features = {
            "net_input": {
                'src_tokens': inputs.input_ids,
                "src_lengths": inputs.length,
                'patch_images': patch_images,
                'patch_masks': patch_masks,
                "prev_output_tokens": prev_output_tokens
            },
            "target": target
        }
        return features
    

class OFADataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # 分布式适配
        local_rank, rank, world_size = world_info_from_env()
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        
    def __getitem__(self, index):
        if type(index) is not int:
            raise NotImplementedError
        else:
            data = self.dataset.__getitem__(index)
            data = TaskProcessor.auto_task(data)
        return data
    
    def __len__(self):
        return len(self.dataset)
    

def _ofa_generator(model, sample, dtype=torch.float16, device='cuda'):
    model.eval()
    model.to(device=device)
    for k, v in sample['net_input'].items():
        sample['net_input'][k] = v.to(device=device, dtype=dtype) if v.dtype == torch.float32 else v.to(device=device)
    with autocast(dtype=dtype):
        with torch.no_grad():
            hypos = task.inference_step(generator, [model], sample)
    hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
    return tokenizer.batch_decode(hypos)


def ofa_generator(model, src_texts: List[str], images: List[Any], dtype=torch.float16, device='cuda'):
    
    src_inputs = tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
    src_tokens = src_inputs.input_ids
    src_lengths = src_inputs.length
    patch_images = torch.stack([image_processor(i) for i in images])
    patch_masks = torch.tensor([True] * len(src_tokens))
    sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
    return _ofa_generator(model, sample, dtype=torch.float16, device='cuda')


class MyCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        bboxes_gt = []
        bboxes_gen = []
        for b in kwargs['eval_dataloader']:
            b['net_input'].pop('prev_output_tokens')
            sbboxes_gt = tokenizer.batch_decode(b.pop('target'))
            sbboxes_gen = _ofa_generator(model, b)
            bboxes_gt += [sbbox_to_bbox(s) for s in sbboxes_gt]
            bboxes_gen += [sbbox_to_bbox(s) for s in sbboxes_gen]
        res = eval_grounding_acc_v2(np.asarray(bboxes_gen), np.asarray(bboxes_gt))
        trainer.log_metrics("eval", res)
        kwargs['metrics']['eval_ap50'] = res['grounding_acc_iou_0.5']

if __name__ == "__main__":
    DEBUG_MODE = True
    DEBUG_MODE = False
    local_rank, rank, world_size = world_info_from_env()

    # load model
    ckpt_path = "/mnt/bn/hri-lq/projects/VLDD/OFA-checkpoints/bk-230402/invig_large_checkpoints/20_3e-5_512_20230325-1127/checkpoint_15_56000.pt"
    ckpt_path = "/mnt/bn/ckpt-lq/vldd/invig_large-3ds_checkpoints/30_1e-4_512_20230404-1407/checkpoint_20_75000.pt"
    ckpt_path = "/mnt/bn/hri-lq/projects/VLDD/OFA-checkpoints/bk-230402/invig_huge_checkpoints/10_3e-5_512_20230326-0235/checkpoint_8_57000.pt"
    ckpt_path = "/mnt/bn/ckpt-lq/vldd/invig_large-3ds_checkpoints/10_3e-5_512_20230414-1106/checkpoint_2_42000.pt"
    generator, model, cfg, task = load_from_pretrain(ckpt_path)
    gens = (generator, model, cfg, task)

    # load processor
    tokenizer = OFATokenizer.from_pretrained("/mnt/bn/hri-lq/projects/VLDD/Link/ofa-pretrain/hf/ofa-large", local_files_only=True, truncation_side="right")
    image_processor = get_image_processor(512)

    # load dataset
    ds_invig     = datasets.load_from_disk(f"/mnt/bn/hri-lq/datasets/hf/invig")['train']
    ds_guesswhat = datasets.load_from_disk(f"/mnt/bn/hri-lq/datasets/hf/guesswhat")['train']
    ds_visdial   = datasets.load_from_disk(f"/mnt/bn/hri-lq/datasets/hf/visdial")['train']

    # logger.info("setup task for train data (1/3)...")
    # ds_invig_q = TaskProcessor.setup_task(ds_invig, "question")
    # ds_invig_a = TaskProcessor.setup_task(ds_invig, "answer")
    ds_invig_g = TaskProcessor.setup_task(ds_invig, "grounding_for_test")
    # logger.info("setup task for train data (2/3)...")
    # ds_guesswhat_q = TaskProcessor.setup_task(ds_guesswhat, "question")
    # ds_guesswhat_a = TaskProcessor.setup_task(ds_guesswhat, "answer")
    ds_guesswhat_g = TaskProcessor.setup_task(ds_guesswhat, "grounding_for_test")
    # logger.info("setup task for train data (3/3)...")
    # ds_visdial_q = TaskProcessor.setup_task(ds_visdial, "question")
    # ds_visdial_a = TaskProcessor.setup_task(ds_visdial, "answer")
    
    # logger.info("setup interleave_datasets for train data...")
    # use_datasets = [ds_invig_q, ds_invig_a, ds_invig_g, ds_guesswhat_q, ds_guesswhat_a, ds_guesswhat_g, ds_visdial_q, ds_visdial_a]
    # prob_datasets = [0.1, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1]
    # dataset = datasets.interleave_datasets(use_datasets, prob_datasets, seed=42, stopping_strategy="all_exhausted")
    
    ds_train = datasets.concatenate_datasets([ds_invig_g, ds_guesswhat_g]).shuffle()
    ds_train = OFADataset(ds_train)

    ds_guesswhat_eval = datasets.load_from_disk(f"/mnt/bn/hri-lq/datasets/hf/guesswhat")['validation']
    ds_valid = TaskProcessor.setup_task(ds_guesswhat_eval, "grounding_for_test").shuffle().select(range(512))
    ds_valid = OFADataset(ds_valid)

    hyper_args = argparse.Namespace(**{
        "per_device_train_batch_size": 3,
        "per_device_eval_batch_size": 6,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "warmup_steps": 1000,
        # "label_smoothing_factor": 0.1,
    })
    setting_args = argparse.Namespace(**{
        "fp16": True,
        "fp16_full_eval": True,
        "dataloader_num_workers": 8,
        "logging_steps": 20 if DEBUG_MODE else 50,
        "report_to": "none" if DEBUG_MODE else "all",
        "output_dir": "./runs-debug" if DEBUG_MODE else "./runs",
        "label_names": ["target"],
        "evaluation_strategy": "steps",
        "eval_steps": 20 if DEBUG_MODE else 200,
        "save_strategy": "steps",
        "save_steps": 40 if DEBUG_MODE else 200,
        "save_total_limit": 20,
        "fsdp": "shard_grad_op",
        # "load_best_model_at_end": True,
        # "ddp_find_unused_parameters": True,
    })

    trainer = transformers.Trainer(
        model=OFAModelWarper(model),
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=DataCollatorForOFA(tokenizer=tokenizer, image_processor=image_processor),
        args=transformers.TrainingArguments(**vars(hyper_args), **vars(setting_args)),
        callbacks=[MyCallback()]
    )
    trainer.train()
