# Author: ***
# Date: 4/2/2023

# builtin library
from dataclasses import dataclass
from typing import Optional, Any, Union, List

# 3rd library
import numpy as np
import transformers


@dataclass
class DataCollator(transformers.DataCollatorForSeq2Seq):
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[Any] = None
    padding: Optional[str] = "longest"
    max_src_length: Optional[int] = 256
    max_tgt_length: Optional[int] = 160
    return_tensors: Optional[str] = "pt"

    def __call__(self, features):
        from .training import world_info_from_env
        """ features: [{'src_text', 'tgt_text', 'image'}, ...] -> {...} """
        import torch
        src_texts = [f['src_text'] for f in features]
        tgt_texts = [f['tgt_text'] for f in features]
        images = [f['image'].convert('RGB') for f in features]

        # max_length longest
        _, rank, world_size = world_info_from_env()
        inputs = self.tokenizer(text=src_texts, return_length=True, max_length=self.max_src_length,
                                padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        labels = self.tokenizer(text=tgt_texts, max_length=self.max_tgt_length,
                                padding=self.padding, truncation=True, return_tensors=self.return_tensors)
        patch_images = torch.Tensor(np.stack(self.image_processor(images)['pixel_values']))
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
            "target": target,
            # ***
            "nsentences": len(inputs.length),
            "ntokens": inputs.length.sum().item()
        }
        return features
