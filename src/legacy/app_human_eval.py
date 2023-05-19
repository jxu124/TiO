from typing import Any, Optional, Union
from PIL import Image
from io import BytesIO
import gradio as gr
import argparse
import datasets
import numpy as np
import requests
import base64
import json
import re
import time
import os

# from chatbot_interface import VDG
# our_model = VDG("/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt")
model_id = "xvlm"
# model_id = "ours"

log_dir = "/mnt/bn/ckpt-lq/vldd/benchmark"
data_info = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(log_dir, "human_eval")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = f"{log_dir}/{model_id}_{data_info}.json"
log_file = f"/mnt/bn/ckpt-lq/vldd/benchmark/human_eval/xvlm_20230515-192335.json"
print(log_file)

test_list = [13, 51, 54, 61, 65, 88, 93, 142, 161, 163, 178, 189, 191, 198, 206, 209, 228, 255, 285, 318, 326, 333, 393, 407, 429, 440, 447, 451, 457, 466, 476, 501, 541, 546, 563, 569, 592, 600, 689, 696, 704, 727, 735, 740, 747, 758, 775, 778, 859, 864, 865, 919, 928, 940, 1034, 1098, 1116, 1130, 1149, 1182, 1206, 1209, 1232, 1236, 1266, 1287, 1301, 1309, 1330, 1354, 1372, 1385, 1429, 1436, 1437, 1442, 1466, 1494, 1508, 1516, 1518, 1554, 1563, 1583, 1650, 1652, 1657, 1698, 1708, 1735, 1751, 1764, 1774, 1780, 1813, 1827, 1888, 1918, 1960, 1992]


def chat_xvlm(task, image, dialog, text, bbox, all_bboxes):
    assert task in ['q', 'a', 'g']
    assert bbox[2] <= 1 and bbox[3] <= 1
    with BytesIO() as f:
        image.convert('RGB').save(f, format='jpeg')
        f.seek(0)
        image_b64 = base64.urlsafe_b64encode(f.read()).decode()

    data = {
        "token": "password",  # 简单校验用的token
        "task": task,
        "text": text,
        "dialog": dialog,
        "bboxes": json.dumps(all_bboxes),
        "ref_bboxes": json.dumps(np.asarray(bbox).reshape(4).tolist()),
        "image": image_b64
    }
    api_url_map = {"q": "invig_test_questioner", "a": "invig_test_oracle", "g": "invig_test_guesser"}
    ret = requests.post(f"http://[fdbd:dc03:9:25:f00::b8]:5010/{api_url_map[task]}", json=data, timeout=10)
    ret = ret.json()['responce']
    return ret

def get_prompt(dialog, text):
    from chatbot_interface import MapFunc
    dialog = [j for i in dialog for j in i] + [text]
    prompt = ["can you specify which region the context describes?", *dialog] + [""]
    prompt = MapFunc.make_text_pair(prompt, human_first=True)[0]
    return prompt

def chat(model_id, image, dialog, text, bbox, all_bboxes):
    print(model_id)
    if model_id == "xvlm":
        text_robot = chat_xvlm("q", image, dialog, text, bbox, all_bboxes)
        if text_robot == "":
            text_robot = chat_xvlm("g", image, dialog, text, bbox, all_bboxes)  # 相对值bbox
            text_robot = np.asarray(text_robot).reshape(4)
    else:
        from chatbot_interface import sbbox_to_bbox
        # todo
        prompt = get_prompt(dialog, text)
        text_robot = our_model.chat([image], [prompt])[0]
        text_robot = re.sub("not yet\.", "", text_robot).strip()
        if "region:" in text_robot:
            text_robot = np.asarray(sbbox_to_bbox(text_robot)).reshape(4)
    return text_robot


ds = datasets.load_from_disk("/mnt/bn/hri-lq/datasets/hf-cache/invig")['test']
all_bboxes = []
raw_image_path = ""

def fn_load_image(index):
    global all_bboxes, raw_image_path
    d = ds[int(index)]
    raw_image_path = d['image_path']
    image_path = raw_image_path.replace("openimages_v1.2", "/mnt/bn/hri-lq/datasets/images/openimages_v1.2")
    image = Image.open(image_path)
    bbox_abs = np.asarray(d['bbox'], dtype=int).tolist()
    all_bboxes = json.loads(d['raw_anns'])
    for ii, _ in enumerate(all_bboxes):
        all_bboxes[ii]['bbox'] = (np.asarray(all_bboxes[ii]['bbox']) / np.asarray([*image.size, *image.size])).tolist()
    # print(all_bboxes)
    return [bbox_abs], (image, [(bbox_abs, "oracle")])


def fn_chatbot(anns, bbox, text, history):
    image = Image.open(anns[0]['name'])
    bbox_gt = bbox[0]
    # text_robot = "which one do you want?"
    text_robot = chat(model_id, image, history, text, [0, 0, 0, 0], all_bboxes)
    if type(text_robot) is not str:
        bbox_pred = np.asarray(text_robot).reshape(4) * np.asarray([*image.size, *image.size])
        bbox_pred = bbox_pred.astype(int).tolist()
        text_robot = json.dumps(bbox_pred)
        anns = (image, [(bbox_gt, "oracle"), (bbox_pred, "predict")])
    else:
        anns = (image, [(bbox_gt, "oracle")])
    # print(type(text_robot))
    # print(anns, history)

    history += [(text, text_robot)]
    return anns, history


def fn_save(image_id, anns, bbox, history):
    image_id = int(image_id)
    image = Image.open(anns[0]['name'])
    bbox_gt = bbox[0]
    try:
        bbox_pred = np.asarray(json.loads(history[-1][-1])).reshape(4).tolist()
    except:
        bbox_pred = [0, 0, 0, 0]
    batch_logs = json.dumps([image_id, history, bbox_gt, bbox_pred, raw_image_path]) + "\n"
    with open(log_file, "a") as f:
        f.writelines(batch_logs)
    print(f"save {image_id} to {log_file}")


def fn_browse():
    try:
        with open(log_file) as f:
            data = f.readlines()
            data = [json.loads(i)[0] for i in data]
        remainder = list(set(test_list) - set(data))
        remainder.sort()
    except:
        data = []
        remainder = test_list
    return f"""
所有需要测试的id号:
{json.dumps(test_list)}

已经保存了的id号: ({len(set(data))})
{json.dumps(data)}

剩余还没测试的id号:
{json.dumps(remainder)}
"""
    

with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=0.8):
                blk_number = gr.Number(label="image_id", info="image_id的取值范围: [0, 1998)")
                btn_load = gr.Button("Load")
                blk_image_bbox = gr.AnnotatedImage(label="image with bbox").style(color_map={"oracle": "#aaaaff", "predict": "#aaffaa"})
                blk_bbox = gr.List(label="GT-bbox", headers=["x1", "y1", "x2", "y2"], datatype="number", row_count=(1, "fixed"), col_count=(4, "fixed"), interactive=False)
                btn_save = gr.Button("Save Dialog")
            with gr.Column(scale=2.4):
                blk_chatbot = gr.Chatbot(label="Chatbot", value=[])
                blk_input = gr.Textbox(label="input", lines=5, value="I love this picture.")
                with gr.Row():
                    btn_clear = gr.Button("Clear")
                    btn_submit = gr.Button("Submit")

    btn_load.click(fn_load_image, inputs=[blk_number], outputs=[blk_bbox, blk_image_bbox], api_name="/load")
    btn_clear.click(lambda: (None, None), None, outputs=[blk_image_bbox, blk_chatbot], queue=False)
    btn_submit.click(fn_chatbot, inputs=[blk_image_bbox, blk_bbox, blk_input, blk_chatbot], outputs=[blk_image_bbox, blk_chatbot], api_name="/predict")
    blk_input.submit(fn_chatbot, inputs=[blk_image_bbox, blk_bbox, blk_input, blk_chatbot], outputs=[blk_image_bbox, blk_chatbot])
    btn_save.click(fn_save, inputs=[blk_number, blk_image_bbox, blk_bbox, blk_chatbot], outputs=[], api_name="/save")

    with gr.Tab("Saved Data"):
        # import random
        # random.seed(42)
        # test = random.sample(range(1998), 100)
        # test.sort()
        # print(test)
        btn_update = gr.Button("刷新")
        blk_browse = gr.Textbox(lines=15)

    btn_update.click(fn_browse, None, outputs=[blk_browse])





    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "--server-name", default="0.0.0.0", type=str)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_name=args.ip, share=args.share)
