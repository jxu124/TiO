# This file is modified from https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/app.py

import argparse
import sys
import os
import random
import re
import json

from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr


def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = False

# ========================================
#             Model Initialization
# ========================================

from tio_core.utils import sbbox_to_bbox
from tio_core.module import OFAModelWarper


def get_sam_predictor():
    global predictor
    if predictor is None:
        # 自动配置sam环境
        try:
            # 初始化sam
            from segment_anything import sam_model_registry, SamPredictor
            sam_checkpoint = "./attachments/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
        except ImportError:
            # os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
            os.system("cd attachments && git clone https://github.com/facebookresearch/segment-anything.git")
            os.system("pip install setuptools==58.0")
            os.system("sudo pip install ./attachments/segment-anything")
            os.system("pip install opencv-python pycocotools matplotlib onnxruntime onnx")
            os.system("cd attachments && wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            predictor = get_sam_predictor()
    return predictor
predictor = None

# ========================================
#             Gradio Setting
# ========================================

def gradio_upload(gr_img):
    # return (image, image_bbox, text_input, chatbot)
    if gr_img is None:
        return None, None, gr.update(placeholder='Please upload your image first', interactive=False), None
    image = Image.open(BytesIO(gr_img)).resize((512, 512), Image.Resampling.BILINEAR)
    return image, (image, []), gr.update(placeholder='Type and press Enter', interactive=True), None


def gradio_chat(image, text_input, chatbot):
    # return (image_bbox, text_input, chatbot)
    def get_bbox(image, text, tag="grounding"):
        if "region:" in text:
            bboxes = sbbox_to_bbox(text) * np.asarray([*image.size, *image.size])
            bboxes = [(b.astype(int).reshape(4).tolist(), f"{tag}_{i}") for i, b in enumerate(bboxes)]
            return bboxes
        return []

    def make_context(dialog, text_input=None):
        context = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in dialog])
        return context if text_input is None else context + f" human: {text_input}"

    # chat
    context = make_context(chatbot, text_input).lower()
    text_response = gradio_predict(image, context)
    bboxes = get_bbox(image, text_response, "output")
    chatbot += [[text_input, text_response]]

    # 计算grounding bbox
    context = f" \n#instruction: which region does the context describe?\n#context: \"{make_context(chatbot, text_input).lower()}\""
    text_response = gradio_predict(image, context)
    bboxes += get_bbox(image, text_response)

    # 生成bbox
    image_bboxes = (image, [*bboxes])
    return image_bboxes, chatbot


def gradio_predict(image, text_input):
    return model.generate([text_input], [image])[0]


def gradio_sam_api(image, bbox):
    predictor = get_sam_predictor()

    predictor.set_image(np.asarray(image))
    input_box = np.asarray(json.loads(bbox), dtype=float).reshape(4) * np.asarray([*image.size, *image.size])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return json.dumps(masks[0].astype(np.uint8).tolist())


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">Demo of TiO</h1>""")

    with gr.Row():
        with gr.Column(scale=0.5):
            gr_image = gr.Image(label='image', visible=False, type="pil")
            gr_image_bbox = gr.AnnotatedImage(label="image with bbox", visible=True).style(color_map={"input": "#aaaaff", "grounding": "#aaffaa"})
            upload_file = gr.File(file_types=['image'], type="binary", label="image")
            upload_button = gr.Button(value="Upload & Chat", interactive=True, variant="primary")

        with gr.Column():
            gr_chatbot = gr.Chatbot(label='chatbot')
            gr_text_input = gr.Textbox(label='text_input', placeholder='Please upload your image first', interactive=False)
            gr_chat_button = gr.Button(value="predict", interactive=True, visible=False)

    upload_button.click(gradio_upload, [upload_file], [gr_image, gr_image_bbox, gr_text_input, gr_chatbot])
    gr_text_input.submit(gradio_chat, [gr_image, gr_text_input, gr_chatbot], [gr_image_bbox, gr_chatbot], 
                         api_name="chat").then(lambda : None, [], [gr_text_input])
    gr_chat_button.click(gradio_predict, [gr_image, gr_text_input], [gr_text_input], api_name="single_chat")

    # for sam
    gr_bbox = gr.Textbox(label='bbox', visible=False)
    gr_mask = gr.Textbox(label='mask', visible=False)
    gr_mask.change(gradio_sam_api, [gr_image, gr_bbox], [gr_mask], api_name="sam")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print('Initializing Chat')
    setup_seeds()
    model = OFAModelWarper(args.ckpt)
    print('Initialization Finished')

    print(f"Run TiO server on 0.0.0.0:{args.port}")
    demo.queue(1).launch(enable_queue=True, server_name="0.0.0.0", server_port=args.port, share=args.share)
