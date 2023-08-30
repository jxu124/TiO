# Author: Xu Jie
# Date: 06/19/23

# builtin-lib
from io import BytesIO
import os
import sys
import time
import argparse
import json
import datetime
import pickle
import pathlib
import tempfile
from typing import List, Tuple
from functools import partial
import random

# 3rd-lib
from PIL import Image as PILImage
import numpy as np
import gradio_client
import gradio as gr
import cv2
import datasets


BASE_PATH = pathlib.Path(os.path.abspath(__file__)).parent


# ===========================================================
# ○ 工具函数
# ===========================================================
def show_mask(image: PILImage.Image, bboxes=None, masks=None, show_id=False, text_size=1) -> PILImage.Image:
    """ 给图片画上mask: 只更改被mask标记部分的rgb值. """
    colors = json.load(open(f"{BASE_PATH}/attachments/color_pool.json"))
    size = image.size
    image = np.asarray(image)
    if bboxes is not None:
        bboxes = np.array(bboxes).reshape(-1, 4)
        for k, bbox in enumerate(bboxes):
            bbox = (np.asarray(bbox) * np.asarray([*size, *size])).astype(int)
            image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), tuple(colors[k]), thickness=2)
        if show_id:
            for k, bbox in enumerate(bboxes):
                bbox = (np.asarray(bbox) * np.asarray([*size, *size])).astype(int)
                image = cv2.putText(image, str(k), tuple(bbox[:2] + np.array([2, 28 * text_size])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2, cv2.LINE_AA)
                image = cv2.putText(image, str(k), tuple(bbox[:2] + np.array([2, 28 * text_size])), cv2.FONT_HERSHEY_SIMPLEX, text_size, tuple(colors[k]), 1, cv2.LINE_AA)

    if masks is not None:
        for k, mask in enumerate(masks):
            mask_color = (mask[..., None] * colors[k][:3]).astype(np.uint8)
            image_mask = cv2.addWeighted(mask_color, 0.5, image * mask[..., None], 0.5, 0)
            image = cv2.add(image * ~mask[..., None], image_mask)
    return PILImage.fromarray(image)


methods = ["TiO", "XVLM", "Seeask2"]
method_urls_default = [["http://10.128.100.214:7864"], ["http://10.128.96.82:7860"], ["http://10.128.98.140:7862"]]
method_options = {
    "method": "seeask",
    "config": "vocab+reclip",
    "detic_url": "http://10.128.98.140:7863",
    "detic_threshold": 0.5
}

def reset_data_cache(data_cache={}):
    data_cache = {
        "init": data_cache.get("init", False),
        "dataset": data_cache.get("ds"),
        "method_urls": data_cache.get("method_urls"),
        "bboxes": [],  # 每一轮的bbox
        "index_list": data_cache.get("index_list"),  # 混合后的index列表
        "index": data_cache.get("index"),  # 当前样本的index
    }
    return data_cache


def func_load_sample(data_files, data_cache, method_urls, set_index=None, set_method=0, seed=42):
    if data_cache is None or set_index is not None:
        if set_index is None:
            set_index = 0
        set_index = int(set_index)
        data_files = [i.name for i in data_files]
        ds = datasets.load_dataset("parquet", data_files=data_files, split="train").shuffle(seed)
        random.seed(seed)
        index_list = [(i, j) for i in range(len(ds)) for j in [set_method]]
        # index_list = [(i, j) for i in range(len(ds)) for j in random.sample(range(len(methods)), len(methods))]
        data_cache = {
            "init": False,
            "dataset": ds,
            "method_urls": method_urls,
            "bboxes": None,  # 每一轮的bbox
            "index_list": index_list,  # 混合后的index列表
            "index": set_index,  # 当前样本的index
        }
    return data_cache


def func_next(data_cache, chatbot, sign="debug_user"):
    # 保存为文件: "[{from_file}][{image_id}][{method_str}].json"
    if data_cache["init"]:
        sample_idx, method_idx = data_cache["index_list"][data_cache["index"]]
        sample_old = data_cache["dataset"][sample_idx]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        item = {
            "image_id": sample_old['sid'],
            "method": methods[method_idx],
            "sign": sign,
            "date": timestamp,
            "dialog": [i for i in chatbot[1:] if not isinstance(i[1], tuple)],
            "bboxes": data_cache['bboxes'],
            "sample": sample_old
        }
        # filename = f"{BASE_PATH}/data/humanchat_0828/[{sign}][{sample_old['sid']}][{methods[method_idx]}].parquet"
        filename = f"{BASE_PATH}/data/humanchat_0828/{sign}-{sample_old['sid']}-{methods[method_idx]}.parquet"
        # import pdb; pdb.set_trace()
        print(f"saving to {filename} ...")
        datasets.Dataset.from_list([item]).to_parquet(filename)
        data_cache["index"] += 1
    data_cache['bboxes'] = []
    data_cache["init"] = True
    # 读取下一个
    sample_idx, method_idx = data_cache["index_list"][data_cache["index"]]
    sample = data_cache["dataset"][sample_idx]
    image = sample['image'].resize([512, 512])
    bbox = np.asarray(sample['objects'][sample['prompt']['target']]['bbox']) * np.asarray([*image.size, *image.size])
    bbox = bbox.reshape(4).astype(int).tolist()
    image_bbox = (image, [(bbox, "your target")])
    # 第一句话对话
    method_str = methods[method_idx]
    data_cache['client'] = gradio_client.Client(data_cache['method_urls'][method_idx][0])
    print(f"Method Select: {method_str}({data_cache['method_urls'][method_idx][0]})")
    data_cache['client'].predict(api_name="/reset")
    system_message = f"""--- System Message ---
image_index: {sample_idx}
image_id: {sample['sid']}
method_id: {method_idx}
你的目标是根据第一轮对话继续和agent对话, 你的对话需要围绕your_target(左下角图片)进行, 直到agent正确地框出了your_target. 推荐使用一些复杂的表达. 请考虑从以下几个方面考验agent:
- 考察模型遮挡物体和复杂环境的理解能力; 
- 考察模型对人类目标的状态\动作\衣着\性别等等的理解能力;
- 考察模型的语言理解能力. 例如: ①(需求) i am hungry ②(OCR) pass me cake with text "W" ③(关系推理) pass me the apple next to banana ④(近似表达/非直接回答) robot: do you mean the banana next to the apple? human: Oh, I want the one on the white plate. ⑤(不常见名词/多义词)pass me a chinese date(给我一个中国枣) 
"""
    chatbot = [[None, system_message], [sample['prompt']['prompt'], None]]
    chatbot, _ = func_chat(data_cache, image, chatbot)
    # gr_image, gr_image_bbox, gr_chatbot, gr_method
    # 图片, 物体bbox, 第一句对话, 方法
    return data_cache, image, image_bbox, chatbot, method_str


# def func_chat(method_client, image, dialog):
def func_chat(data_cache, image, dialog):
    text_input = dialog[-1][0]
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        image.save(f)
        try:
            chat_response = data_cache['client'].predict(f.name, text_input, json.dumps(method_options), api_name="/chat")
        except BaseException:
            import traceback
            traceback.print_exc()
            print("send:", f.name, text_input, json.dumps(method_options))
        chat_response = json.loads(chat_response)
    # import pdb; pdb.set_trace()
    dialog[-1][1] = chat_response['text_response']
    if dialog[-1][1] == "":
        dialog[-1][1] = None
    if chat_response['bboxes'] is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            show_mask(image, chat_response['bboxes']).save(f)
        dialog += [[None, (f.name,)]]
        data_cache['bboxes'] += [np.asarray(chat_response['bboxes']).reshape(-1, 4)[0]]
    else:
        data_cache['bboxes'] += [None]
    print("data_cache['bboxes']", data_cache['bboxes'])
    return dialog, gr.update(interactive=not chat_response['done'])


# ===========================================================
# ○ main gradio
# ===========================================================
def main(port: int = None, seed: int = 42):
    def make_interactive_if(set_method, seed):
        # 只需要输入文字, 然后对话完了点下一个
        with gr.Blocks() as demo:
            # gr.Markdown("""<h1 align="center">Evaluation Demo</h1>""")

            eval_files = [
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_1.parquet",
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_2.parquet",
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_3.parquet",
            ]
            gr_cache = gr.State()
            # gr_method_client = gr.State()
            with gr.Row():
                with gr.Column(scale=0.3):
                    # 方法应该自动切换
                    with gr.Accordion("debug", open=False):
                        gr_files = gr.Files(label="validation_files", value=eval_files, interactive=False)
                        gr_method_urls = gr.List(label="method_urls", value=method_urls_default)
                        gr_method = gr.Dropdown(label="agent", choices=methods, value=methods[set_method], type="index", visible=True)
                        gr_set_index = gr.Text(label="set_index")
                    gr_image = gr.Image(label="image", interactive=False, type="pil")
                    gr_image_bbox = gr.AnnotatedImage(label="your_target", interactive=False)
                    gr_sign = gr.Text(label="签名", value="debug_user")
                    gr_button= gr.Button("next", variant="primary")

                with gr.Column():
                    gr_chatbot = gr.Chatbot(label='chatbot', height=768)
                    gr_text_input = gr.Textbox(label='text_input', interactive=False)

                # 更改文件后重置状态
                gr_files.update(lambda : None, [], [gr_cache])
                # 下一个按钮 - 获取新数据 -> 显示数据信息
                func_load = partial(func_load_sample, seed=seed, set_method=set_method)
                gr_button.click(func_load, [gr_files, gr_cache, gr_method_urls], [gr_cache])\
                    .then(func_next, [gr_cache, gr_chatbot, gr_sign], [gr_cache, gr_image, gr_image_bbox, gr_chatbot, gr_method])\
                    .then(lambda : gr.update(value=None, interactive=True), [], [gr_text_input])
                # 与agent对话
                gr_text_input.submit((lambda text, dialog: (None, dialog + [[text, None]])), [gr_text_input, gr_chatbot], [gr_text_input, gr_chatbot])\
                    .then(func_chat, [gr_cache, gr_image, gr_chatbot], [gr_chatbot, gr_text_input])

                # 跳转到指定样本
                gr_set_index.submit(func_load, [gr_files, gr_cache, gr_method_urls, gr_set_index], [gr_cache])\
                    .then(func_next, [gr_cache, gr_chatbot, gr_sign], [gr_cache, gr_image, gr_image_bbox, gr_chatbot, gr_method])\
                    .then(lambda : gr.update(value=None, interactive=True), [], [gr_text_input])
        return demo

    def make_score_if():
        ds = datasets.load_dataset("parquet", data_files=[
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_1.parquet",
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_2.parquet",
                "/mnt/bn/hri-lq/projects/RoboTiO/evaluation/fin/testset_3.parquet",
            ], split="train").shuffle(42)
        score = json.load(open("data/humanchat_score.json", "r"))

        def random_methods_vs(methods_vs):
            return [methods_vs[i] for i in np.random.choice(a=len(methods_vs), size=len(methods_vs), replace=False)]

        def load_dialog(user, image_id, method):
            def get_image_path(image):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    image.save(f)
                return f.name

            filepath = f"data/humanchat_0828/{user}-{image_id}-{method}.parquet"
            ds = datasets.Dataset.from_parquet(filepath)
            d = ds[0]
            with BytesIO(d['sample']['image']['bytes']) as f:
                image = PILImage.open(f).convert("RGB")
            dialog = [[(get_image_path(image),), None]]
            print(filepath)
            print("d['bboxes']:", d['bboxes'])
            for d, bbox in zip(d['dialog'], d['bboxes'][-len(d['dialog']):]):
                dialog += [d]
                if bbox is not None:
                    dialog += [[None, (get_image_path(show_mask(image, bbox)),)]]
            return dialog
        # dialog = load_dialog("debug_user", "obj365.254.84", "TiO")

        # methods = ["TiO", "XVLM", "Seeask2"]
        methods_vs = [["TiO", "XVLM"], ["TiO", "Seeask2"]]
        eval_samples = [(image_id, method_vs) for image_id in ds['sid'] for method_vs in random_methods_vs(methods_vs)]

        def get_next_sample(user, state, label):
            # 计算保存评分
            if state is not None:
                image_id = state['image_id']
                method_vs = state['method_vs']
                if method_vs[0] == "TiO":
                    method = method_vs[1]
                else:
                    method = method_vs[0]
                    if isinstance(label, bool):
                        label = not label
                if method == "XVLM":
                    score[(image_id, user)] = {
                        '1vs2': label
                    }
                elif method == "Seeask2":
                    score[(image_id, user)] = {
                        '1vs3': label
                    }
                print(score)
                # score = json.dump(open("data/humanchat_score.json", "w"))
                state['index'] += 1
            else:
                state = {'index': 0}

            # 获取下一个样本
            print(state['index'], eval_samples[state['index']])
            state['image_id'], state['method_vs'] = eval_samples[state['index']]
            state['method_vs'] = random_methods_vs(state['method_vs'])
            method1, method2 = state['method_vs']
            dialog1 = load_dialog(user, state['image_id'], method1)
            dialog2 = load_dialog(user, state['image_id'], method2)
            return state, dialog1, dialog2

        from functools import partial
        left_click = partial(get_next_sample, label=True)
        bad_click = partial(get_next_sample, label="bad")
        good_click = partial(get_next_sample, label="good")
        right_click = partial(get_next_sample, label=False)

        # 只需要输入文字, 然后对话完了点下一个
        with gr.Blocks() as demo:

            gr_state = gr.State()
            gr_label = gr.Text(label="数据标签(username)", value="debug_user")
            with gr.Row():
                gr_chatbot1 = gr.Chatbot(label='method A', height=768)
                gr_chatbot2 = gr.Chatbot(label='method B', height=768)
            with gr.Row():
                gr_btn_left = gr.Button("👈左边更好")
                gr_btn_bad = gr.Button("👎都不好")
                gr_btn_good = gr.Button("👍都很好")
                gr_btn_right= gr.Button("👉右边更好")

            gr_btn_left.click(  left_click, [gr_label, gr_state], [gr_state, gr_chatbot1, gr_chatbot2])
            gr_btn_bad.click(    bad_click, [gr_label, gr_state], [gr_state, gr_chatbot1, gr_chatbot2])
            gr_btn_good.click(  good_click, [gr_label, gr_state], [gr_state, gr_chatbot1, gr_chatbot2])
            gr_btn_right.click(right_click, [gr_label, gr_state], [gr_state, gr_chatbot1, gr_chatbot2])

        return demo

    try:
        demo = gr.TabbedInterface([make_interactive_if(i, seed) for i in range(3)] + [make_score_if()], tab_names=[f"method {i+1}" for i in range(3)] + ["eval"], title="Evaluation")
        demo.queue(12).launch(server_name="0.0.0.0", server_port=port)
    finally:
        print('Gradio exit...')


if __name__ == "__main__":
    import fire
    fire.Fire(main)
