
from typing import Optional, Any, Union, List
import logging
import os
import torch
import numpy as np

from ..utils import TiOConfig
from .. import path_ofa, path_training_config, path_ckpt

logger = logging.getLogger(__name__)


class OFAModelWarper(torch.nn.Module):
    def __init__(self, load_from=None):
        super().__init__()
        # 自动配置ckpt
        if load_from is None:
            load_from = path_ckpt
        if not os.path.exists(load_from):
            from huggingface_hub import hf_hub_download
            load_from = hf_hub_download("jxu124/tio-checkpoint-zoo", "checkpoint.1_0_0.pt", subfolder="ckpts")
        self.gens = self.load_from_pretrain(load_from)
        self.model = self.gens[1]
        self.tokenizer, self.image_processor = self.task.tio_config.tokenizer, self.task.tio_config.image_processor
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
        if "target" in batch:
            target = self.model.get_targets(batch, net_output)
            # .softmax(dim=1)
            loss = torch.nn.functional.cross_entropy(lprobs.view(-1, 59457), target.view(-1),
                                                     ignore_index=self.tokenizer.pad_token_id, label_smoothing=0.1)
        else:
            loss = None
        return Seq2SeqLMOutput(loss=loss, logits=lprobs)

    @staticmethod
    def load_from_pretrain(load_from):
        from utils import checkpoint_utils
        logger.info(f"Loading from {load_from}...")
        model_overrides = {"bpe_dir": f"{path_ofa}/utils/BPE", "config_yaml": path_training_config}
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([load_from], model_overrides)
        model = models[0]
        model.eval()
        # model.prepare_for_inference_(cfg)
        cfg.generation.max_len_b = 1024
        generator = task.build_generator(models, cfg.generation)
        logger.info("Load Completed.")
        return generator, model, cfg, task

    def generate_origin(self, sample, dtype=torch.float16, device='cuda'):
        from torch.cuda.amp import autocast
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
        sample = self.make_sample(src_texts, images)
        return self.generate_origin(sample, dtype=torch.float16, device='cuda')

    def make_sample(self, src_texts: List[str], images: List[Any]):
        src_inputs = self.tokenizer(src_texts, return_length=True, padding=True, return_tensors="pt")
        src_tokens = src_inputs.input_ids
        src_lengths = src_inputs.length
        patch_images = torch.Tensor(np.stack(self.image_processor(images)['pixel_values']))

        patch_masks = torch.tensor([True] * len(src_tokens))
        sample = {"net_input": {"src_tokens": src_tokens, 'src_lengths': src_lengths, 'patch_images': patch_images, 'patch_masks': patch_masks}}
        return sample

    def bin_check(self, image, text, prompt="is the object human want in the image? yes or no?"):
        text = f" {prompt} #context: \"{text}\""
        text_response = self.generate([text], [image])[0]
        sample = self.make_sample([text], [image])
        sample['net_input']['prev_output_tokens'] = sample['net_input']['src_tokens'][..., :1]
        for k, v in sample['net_input'].items():
            sample['net_input'][k] = v.to(device="cuda", dtype=torch.float16) if v.dtype == torch.float32 else v.to(device="cuda")
        output = self.forward(**sample)
        logits = output.logits

        id_yes, id_no = self.tokenizer.convert_tokens_to_ids("Ġyes"), self.tokenizer.convert_tokens_to_ids("Ġno")
        logits = logits[0, 0].view(59457)[[id_yes, id_no]]
        prob = logits.softmax(dim=0)
        return {"prob_yes": prob[0].item(), "raw_response": text_response}
