# This file is modified from https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/app.py

import argparse
import sys
import os
import random
import re
import pathlib

from io import BytesIO
from PIL import Image
import numpy as np
import yaml
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

SHARED_UI_WARNING = '''### [NOTE] It is possible that you are waiting in a lengthy queue.

You can duplicate and use it with a paid private GPU.
'''

print('Initializing Chat')
CONV_VISION = {}

# ######################### configuration ##########################
from tio_core.utils import sbbox_to_bbox
from tio_core.module import OFAModelWarper
model = OFAModelWarper()

print('Initialization Finished')

class DummyModel():
    image = None
    user_bbox = []
    user_message = ""
    dialog = None

    def upload_img(self, gr_img, chat_state, img_list):
        self.image = gr_img
        return ""

    def ask(self, user_message, dialog, chat_state):
        self.user_message = user_message
        self.dialog = dialog
        self.src_text = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in self.dialog])
        self.src_text += f" human: {self.user_message}"

    def answer(self, conv, img_list, max_new_tokens=300, max_length=2000, prompt_style=0):
        if prompt_style == 1:
            self.src_text = f" \n#instruction: can you specify which region the context describes?\n#context: \"{self.src_text}\""
        elif prompt_style == 2:
            self.src_text = f" \n#instruction: which region does the context describe?\n#context: \"{self.src_text}\""
        agent_message = model.generate([self.src_text.lower()], [self.image])
        agent_message = [re.sub("not yet\.", "", m).strip() for m in agent_message]
        return agent_message

chat = DummyModel()

# ========================================
#             Gradio Setting
# ========================================

def upload_img(gr_img, text_input, option):
    if gr_img is None:
        return None, None, gr.update(interactive=True)
    chat_state = None
    img_list = []
    image = Image.open(BytesIO(gr_img)).resize((512, 512), Image.Resampling.BILINEAR)
    llm_message = chat.upload_img(image, chat_state, img_list)
    anns = (image, [])
    chatbot = gradio_option(option)
    return chatbot, anns, gr.update(interactive=True, placeholder='Type and press Enter')


def gradio_ask(user_message, chatbot, chat_state):
    user_message = user_message.strip()
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chatbot, chat_state)
    chatbot = chatbot + [[user_message, None]]

    if "region:" in user_message:
        bbox = sbbox_to_bbox(user_message) * np.asarray([*chat.image.size, *chat.image.size])
        chat.user_bbox = [(bbox[-1].astype(int).reshape(4).tolist(), "input")]
    # else:
    #     chat.user_bbox = chat.user_bbox[:1]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, option):
    def get_bbox(message):
        if "region:" in message:
            bbox = sbbox_to_bbox(message) * np.asarray([*chat.image.size, *chat.image.size])
            bbox = [(b.astype(int).reshape(4).tolist(), f"output_{i}") for i, b in enumerate(bbox)]
        else:
            bbox = []
        return bbox
    prompt_style = 1 if option == 2 else 0
    llm_message = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=300, max_length=2000, prompt_style=prompt_style)[0]
    bbox = get_bbox(llm_message)
    if option == 1:  # "Freeform (w/ Grounding)":
        chat.src_text += f" agent: {llm_message}\n"
        llm_message_gnding = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=300, max_length=2000, prompt_style=2)[0]
        bbox = get_bbox(llm_message_gnding)

    if "region:" in llm_message:
        index = llm_message.index("region:")
        llm_message = llm_message[:index]
    anns = (chat.image, [*chat.user_bbox, *bbox])
    chatbot[-1][1] = llm_message.strip()
    return anns, chatbot, chat_state, img_list


def gradio_option(option):
    if option == 2:
        chatbot = [("", "which one do you want?")]
    # elif option == "Oracle":
    #     dialog = ["answer the question based on the region.", *dialog]
    #     prompt = DataPrompt.make_text_pair(dialog, human_first=True)
    #     chatbot = [("can you see the image?", "yes, which one do you want?")]
    else:
        chatbot = None
    return chatbot


title = """<h1 align="center">Demo of TiO</h1>"""
description = """<h3>This is the demo of TiO, Upload your images and start chatting!</h3>"""


# TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    # gr.Markdown(SHARED_UI_WARNING)
    # gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            # image = gr.Image(type="pil")
            image_bbox = gr.AnnotatedImage(label="image with bbox", visible=True).style(color_map={"user input": "#aaaaff", "predict": "#aaffaa"})
            upload_file = gr.File(file_types=['image'], type="binary", label="image")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='TiO')
            option = gr.Dropdown([
                "Freeform", 
                "Freeform (w/ Grounding)",
                "Questioner/Guesser", 
                # "Oracle"
            ], label="PromptStyle", type="index", value="Freeform (w/ Grounding)")
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

    upload_button.click(upload_img, [upload_file, text_input, option], [chatbot, image_bbox, text_input])
    option.change(gradio_option, [option], [chatbot])

    text_input.submit(
        gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]
    ).then(
        gradio_answer, [chatbot, chat_state, img_list, option], [image_bbox, chatbot, chat_state, img_list]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(enable_queue=True, server_name="0.0.0.0", share=args.share)
