from typing import Any, Optional, Union
from PIL import Image
import gradio as gr
import argparse
import numpy as np

from chatbot_interface import VDG, bbox_to_sbbox, sbbox_to_bbox, MapFunc


model = VDG("/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt")


def vischatg_fn(image, input):
    output = model.chat([image], [input])[0]
    return output


demo = gr.Interface(
    vischatg_fn,
    inputs=[
        gr.Image(label="image", type='pil'),
        gr.Textbox(label="input", lines=5, value=" \n#instruction: what does the image describe?")
    ],
    outputs=gr.Textbox(label="output", lines=5)
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "--server-name", default="0.0.0.0", type=str)
    args = parser.parse_args()
    demo.launch(server_name=args.ip)
