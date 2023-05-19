
from typing import Optional, Any, Union, List
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)


class OFAModelWarper(torch.nn.Module):
    def __init__(self, ckpt_path, ofa_path):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ofa_path = ofa_path
        self.gens = self.load_from_pretrain(ckpt_path, ofa_path)
        _, self.model, _, task = self.gens
        self.tokenizer, self.image_processor = task.tokenizer, task.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id
        # for huggingface trainer
        from transformers import PretrainedConfig
        self.config = PretrainedConfig(hidden_size=self.model.decoder.embed_tokens.weight.shape[1])

    @property
    def task(self):
        return self.gens[3]

    @property
    def dtype(self):
        return self.model.decoder.embed_tokens.weight.dtype

    def forward(self, **batch):
        from transformers.modeling_outputs import Seq2SeqLMOutput
        net_output = self.model(**batch['net_input'])
        lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
        target = self.model.get_targets(batch, net_output)
        loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1),
                                                 ignore_index=self.pad_token_id, label_smoothing=0.1)
        return Seq2SeqLMOutput(loss=loss, logits=lprobs)

    @staticmethod
    def load_from_pretrain(ckpt_path, ofa_path):
        try:
            from utils import checkpoint_utils
        except ImportError:
            raise ImportError("Cannot import 'utils', please set OFA path to sys.path.")
        logger.info("开始载入model参数...")
        model_overrides = {"bpe_dir": f"{ofa_path}/utils/BPE"}
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], model_overrides)
        model = models[0]
        logger.info("载入ema参数...")
        # model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        logger.info("设置模型...")
        model.eval()
        # model.prepare_for_inference_(cfg)
        generator = task.build_generator(models, cfg.generation)
        logger.info("完成！")
        return generator, model, cfg, task

    def generate_origin(self, sample, dtype=torch.float16, device='cuda'):
        from torch.cuda.amp import autocast
        # tokenizer image_processor
        generator, model, cfg, task = self.gens
        model.eval()
        model.to(device=device)
        for k, v in sample['net_input'].items():
            sample['net_input'][k] = v.to(device=device, dtype=dtype) if v.dtype == torch.float32 else v.to(device=device)
        with autocast(dtype=dtype):
            with torch.no_grad():
                hypos = task.inference_step(generator, [model], sample)
        hypos = [hypos[i][0]["tokens"] for i in range(len(hypos))]
        return self.tokenizer.batch_decode(hypos, skip_special_tokens=True)

    def generate(self, src_texts: List[str], images: List[Any], dtype=torch.float16, device='cuda'):
        src_inputs = self.tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
        src_tokens = src_inputs.input_ids
        src_lengths = src_inputs.length
        patch_images = torch.Tensor(np.stack(self.image_processor(images)['pixel_values']))

        patch_masks = torch.tensor([True] * len(src_tokens))
        sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
        return self.generate_origin(sample, dtype=torch.float16, device='cuda')
