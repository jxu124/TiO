
# TiO - An Interactive Visual Grounding Model
TiO is an Interactive Visual Grounding Model for Disambiguation. (WIP)

## Model and Pretrain Weights
https://huggingface.co/jxu124/TiO


## Usage
```python
import os
os.system("pip install transformers")

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
from io import BytesIO
import torch
import requests

# Load model, tokenizer, image_processor
tokenizer = AutoTokenizer.from_pretrained("jxu124/TiO", use_fast=False)
image_processor = AutoImageProcessor.from_pretrained("jxu124/TiO")
model = AutoModel.from_pretrained("jxu124/TiO", trust_remote_code=True)
model = model.to(torch.float16).cuda()  # It will be faster when using float16.

# Prepare example
image = Image.open(BytesIO(requests.get("http://images.cocodataset.org/val2014/COCO_val2014_000000429913.jpg").content))
text = " #instruction: guess what i want? \n #context: \"human: look that man in white! \""

# Inference
with torch.no_grad():
    pt_txt = tokenizer([text], return_tensors="pt").input_ids.cuda()
    pt_img = image_processor([image], return_tensors="pt").pixel_values.to(torch.float16).cuda()
    gen = model.generate(pt_txt, patch_images=pt_img, top_p=0.5, do_sample=True, no_repeat_ngram_size=3, max_length=256)
print(tokenizer.batch_decode(gen, skip_special_tokens=True))
# e.g. [' is he the one who just threw the ball?']  # Due to the generator, different results may be output
```
