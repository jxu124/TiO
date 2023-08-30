# Author: Xu Jie
# Date: 06/19/23

# builtin-lib
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

# 3rd-lib
from PIL import Image as PILImage
import numpy as np
import gradio as gr
import cv2


BASE_PATH = pathlib.Path(os.path.abspath(__file__)).parent


# ===========================================================
# ○ module define
# ===========================================================
from tio_core.utils import sbbox_to_bbox, bbox_to_sbbox
from tio_core.module import OFAModelWarper
class ChatTiO():
    def __init__(self):
        self.model = OFAModelWarper()
        self.reset()

    def reset(self):
        # [[人类(answer), 机器人(questioner)], ...]
        self.dialog = []

    def chat(self, image: dict, text: str, options: str = "{}") -> str:
        assert isinstance(image, dict)
        assert isinstance(text, str)
        image = image.get("rgb")
        # image_depth = image.get("depth")
        options = json.loads(options)

        # return (image_bbox, text_input, chatbot)
        def get_bbox(text, tag="grounding"):
            if "region:" in text:
                bboxes = sbbox_to_bbox(text)
                # bboxes = [(b.astype(int).reshape(4).tolist(), f"{tag}_{i}") for i, b in enumerate(bboxes)]
                return bboxes.reshape(-1, 4).tolist()
            return []

        def make_context(dialog, text_input=None):
            context = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in dialog])
            return context if text_input is None else context + f" human: {text_input}"

        task = options.get("task", "q")
        bboxes = options.get("bboxes")
        assert task in ["q", "g", "a"]

        chatbot = self.dialog
        if task == "q":
            # chat
            text = text.strip(".") + "."
            context = make_context(chatbot, text).lower()
            context = f"#instruction: guess what i want?\n#context: \"{context}\""
            text_response = self.model.generate([context], [image])[0].replace("region:", "").strip()
            # 对话超过6轮 或者 已经说过相同的话了
            qs = [i[1] for i in chatbot]
            if len(chatbot) >= 6 or text_response in qs:
                text_response = "OK, I see."
            chatbot += [[text, text_response]]
            done = text_response.endswith(".") or text_response.endswith("!")

            # 计算grounding bbox
            if "which" not in text_response and "what" not in text_response and "where" not in text_response:
                context = f" #instruction: which region does the context describe?\n#context: \"{make_context(chatbot, text).lower()}\""
                sbbox_response = self.model.generate([context], [image])[0].strip()
                bboxes = get_bbox(sbbox_response)
        elif task == "a":
            # chat
            sbbox = bbox_to_sbbox(bboxes)
            context = make_context(chatbot, text).lower()
            context = f" \n#instruction: answer the question based on the region.\n#context: \"{context}\" \n#region: {sbbox}"
            text_response = self.model.generate([context], [image])[0].strip()
            bboxes = None
            done = False
        elif task == "g":
            # 计算grounding bbox
            context = f" \n#instruction: which region does the context describe?\n#context: \"{make_context(chatbot, text).lower()}\""
            text_response = self.model.generate([context], [image])[0].strip()
            bboxes = get_bbox(text_response)
            done = False
        self.dialog = chatbot
        return {
            "text_response": text_response,
            "bboxes": bboxes,
            "done": done
        }


DEFAULT_OPTIONS = {}


# ===========================================================
# ○ main
# ===========================================================
if __name__ == "__main__":
    def main(port: int = None):
        from AppBase import main as gr_main
        return gr_main(model=ChatTiO(), default_options=DEFAULT_OPTIONS, title="TiO Demo", port=port)
    import fire
    fire.Fire(main)
