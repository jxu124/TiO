# Author: Xu Jie
# Date: 06/19/23

# builtin-lib
import os
import sys
import json
import pickle
import pathlib
import tempfile
from typing import List, Tuple, Callable, Any

# 3rd-lib
from PIL import Image as PILImage
import numpy as np
import gradio as gr
import cv2


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
        bboxes = np.asarray(bboxes).reshape(-1, 4)
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


# ===========================================================
# ○ main gradio
# ===========================================================
def main(model: Any, default_options: str = "{}", title="Base Demo", port: int = None):
    def chat(image_file, text, options):
        if image_file.name.endswith(".pkl"):
            image = pickle.load(open(image_file.name, "rb"))
        else:
            image = {"rgb": PILImage.open(image_file.name).convert("RGB")}
        ret = model.chat(image, text, options)
        return json.dumps(ret)

    def gr_func_chat(image_file, text, options, chatbot):
        if image_file.name.endswith(".pkl"):
            image = pickle.load(open(image_file.name, "rb"))
        else:
            image = {"rgb": PILImage.open(image_file.name)}
        ret = model.chat(image, text, options)
        text_response = ret.get("text_response")
        bboxes = ret.get("bboxes")
        done = ret.get("done", False)
        chatbot[-1][1] = text_response
        if bboxes is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_bbox:
                show_mask(image['rgb'], bboxes).save(image_bbox)
            chatbot += [[None, (image_bbox.name,)]]
        return gr.update(value=None, interactive=not done), text_response, chatbot

    def get_color_path(fileblock):
        image_path = fileblock.name
        if image_path.endswith(".pkl"):
            image = pickle.load(open(image_path, "rb"))['rgb']
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_tmp_file:
                if max(image.size) > 1024:
                    image = image.resize(np.asarray(image.size) * 1024 // max(image.size))
                image.save(image_tmp_file)
            image_path = image_tmp_file.name
        return image_path

    with gr.Blocks() as demo:
        gr.Markdown(f"""<h1 align="center">{title}</h1>""")

        api_chat = gr.Button(visible=False)
        state_response = gr.Text(visible=False, label="output")
        with gr.Row():
            with gr.Column(scale=0.2):
                gr_file_block = gr.File(file_types=['image', ".pkl"], label="image")
                gr_reset_button = gr.Button(value="Reset", interactive=True, variant="primary")
                gr_options = gr.Text(value=json.dumps(default_options, indent=4), max_lines=5, label="options")

            with gr.Column():
                gr_chatbot = gr.Chatbot(label='chatbot', height=768)
                gr_text_input = gr.Textbox(label='text_input', interactive=False)

        # [调用]api:reset
        #   1. 调用reset; 2. 清空输入框和chatbot
        gr_reset_button.click(model.reset, [], [], api_name="reset")\
            .then((lambda: (gr.update(value=None, interactive=False), None, None)), [], [gr_text_input, gr_chatbot, gr_file_block])
        # [调用]api:chat (file, text, options) -> text_out
        #   1. 调用chat; 2. 更新chatbot显示
        api_chat.click(chat, [gr_file_block, gr_text_input, gr_options], [state_response], api_name="chat")
        gr_text_input.submit((lambda text, chatbot: chatbot + [[text, None]]), [gr_text_input, gr_chatbot], [gr_chatbot])\
            .then(gr_func_chat, [gr_file_block, gr_text_input, gr_options, gr_chatbot], [gr_text_input, state_response, gr_chatbot])
        # [手动]上传图片 -> 实时显示
        #   1. 使能fileblock,并在chatbot上显示图片;
        gr_file_block.upload(
            (lambda fileblock, chatbot: (gr.update(value=None, interactive=True), chatbot + [[(get_color_path(fileblock),), None]])),
            [gr_file_block, gr_chatbot], [gr_text_input, gr_chatbot]
        ).then(model.reset, [], [])
        #   1. 关闭fileblock,清空chatbot;
        gr_file_block.clear(
            (lambda: (gr.update(value=None, interactive=False), None)), 
            [], [gr_text_input, gr_chatbot]
        )

    try:
        demo.queue(1).launch(server_name="0.0.0.0", server_port=port)
    finally:
        print('Gradio exit...')


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    pass
