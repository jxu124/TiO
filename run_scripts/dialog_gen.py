import sys
sys.path.insert(0, "/mnt/bn/hri-lq/projects/TiO/attachments/datasets/src")
import datasets

import os
import pathlib

BASE_PATH = pathlib.PosixPath(os.path.abspath(__file__)).parent.parent

import sys
sys.path.append(str(BASE_PATH))

from tqdm.auto import tqdm
from functools import partial
import random
import json
import datasets
import numpy as np

from tio_core.utils import bbox_to_sbbox, sbbox_to_bbox


def get_model(url="http://127.0.0.1:7860/"):
    import tempfile
    import gradio_client
    client = gradio_client.Client(url)

    def chat(image, text):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_tmp_file:
            image.resize([512, 512]).save(image_tmp_file)
            text_out = client.predict(image_tmp_file.name, text, api_name="/single_chat")
        return text_out
    return chat


def xywh2xyxy(bbox):
    bbox[..., 2:] += bbox[..., :2]
    return bbox


def get_context(dialog=[], text_user=""):
    src_text = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in dialog])
    if text_user:
        src_text += f" human: {text_user}"
    return src_text


def question_and_answer(model, image, bbox, dialog=[]):
    w, h = image.size
    sbbox = bbox_to_sbbox(bbox, w, h)
    
    prompt_human = " answer the question." if dialog else "state your query."
    prompt_agent = " guess what i want."
    context = f" \n#context: \"{get_context(dialog)}\"" if dialog else ""
    
    text_user = model(image, f"{prompt_human} {context} \n#region: {sbbox}").strip()
    text_agent = model(image, f"{prompt_agent} \n#context: \"{get_context(dialog, text_user)}\"").strip()
    dialog += [(text_user, text_agent)]
    
    is_end = text_agent.startswith("yes") or text_agent.startswith("that")
    text_grounding = model(image, f" which region does the context describe? \n#context: \"{get_context(dialog)}\"").strip()
    
    bbox_pred = sbbox_to_bbox(text_grounding, w, h)
    return dialog, bbox_pred, is_end


def select_bbox_random(item):
    annotations = item['annotations']

    bbox_id = random.randint(0, len(annotations)-1)
    bbox_xywh = annotations[bbox_id]['bbox']
    bbox = xywh2xyxy(np.asarray([bbox_xywh]))
    return bbox_id, bbox


def select_bbox_by_area(item):
    # 采样权重：面积的0.5次方
    annotations = item['annotations']

    w = np.sqrt([a['area'] for a in annotations]).tolist()
    bbox_id = random.choices(list(range(len(annotations))), weights=w, k=1)[0]
    bbox_xywh = annotations[bbox_id]['bbox']
    bbox = xywh2xyxy(np.asarray([bbox_xywh]))
    return bbox_id, bbox


def gen_dialog(item, model):
    image = item['image_raw']
    image_id = item['image']['image_id']
    # bbox_id, bbox = select_bbox_random(item)
    bbox_id, bbox = select_bbox_by_area(item)
    _id = f"{image_id}.{bbox_id}"

    dialog = []
    bbox_pred_list = []
    for i in range(6):
        dialog, bbox_pred, is_end = question_and_answer(model, image, bbox, dialog)
        bbox_pred_list += [bbox_pred]
        if is_end and i > 3:
            break
            
    return {
        "_id": _id,
        "image_id": image_id,
        "dialog": dialog,
        "bbox_gt": bbox.round(1).tolist(),
        "bbox_preds": np.asarray(bbox_pred_list).round(1).tolist()
    }


def load_ds(start=0, end=1):
    data_files = [f"hdfs:///home/byte_ailab_roboticsresearch/user/xujie/hf_datasets/sam/sam-{i:06d}-of-010000.parquet" for i in range(start, end)]
    ds = datasets.load_dataset("parquet", split="train", data_files=data_files, streaming=True)
    return ds


if __name__ == "__main__":
    # python3 run_scripts/dialog_gen.py --index 0 --save_to /mnt/bn/llmdata/xj/tio_gen/sam
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:7860/")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--save_to", type=str)
    args = parser.parse_args()

    model = get_model(args.server)
    ds = load_ds(args.start, args.end)
    save_dir = args.save_to

    for item in tqdm(ds):
        try:
            data = gen_dialog(item, model)
            save_path = f"{save_dir}/{data['_id']}.json"
            print(data)

            with open(save_path, 'w') as f:
                json.dump(data, f)
        except:
            pass
