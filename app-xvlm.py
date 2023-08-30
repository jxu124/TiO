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


BASE_PATH = pathlib.Path(os.path.abspath(__file__)).parent


# ===========================================================
# ○ module define
# ===========================================================
class ChatXVLM():
    def __init__(self):
        AVISION_PATH = "/mnt/bn/hri-lq/projects/alpha_vision"
        os.chdir(AVISION_PATH)
        sys.path.append(AVISION_PATH)
        sys.path.append(f"{AVISION_PATH}/scripts/invig")

        from server_invig_xvlm_cls import parse_args, init_model, Compose, ask, guess, answer
        args = parse_args()
        guesser_model, oracle_model, questioner_model, cfg = init_model(args)
        pipeline = Compose([
            dict(
                type='MultiScaleFlipAug',
                img_scale=(cfg.image_res, cfg.image_res),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='Normalize', **cfg.img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                ]
            )
        ])

        from functools import partial
        self.func_ask = partial(ask, task="q", model=questioner_model, cfg=cfg, pipeline=pipeline)
        self.func_guess = partial(guess, task="g", model=guesser_model, cfg=cfg, pipeline=pipeline)
        self.func_answer = partial(answer, task="a", model=oracle_model, cfg=cfg, pipeline=pipeline)
        self.detic = None

        self.reset()

    def reset(self):
        # [[人类(answer), 机器人(questioner)], ...]
        self.dialog = []

    def chat(self, image: dict, text: str, options: str = "{}") -> str:

        assert isinstance(image, dict)
        assert isinstance(text, str)
        image_color = image.get("rgb")
        # image_depth = image.get("depth")
        options = json.loads(options)

        # options: detic_url task bboxes dialog
        detic_url = options.get("detic_url", DEFAULT_OPTIONS['detic_url'])
        detic_threshold = options.get("detic_threshold", DEFAULT_OPTIONS['detic_threshold'])
        task = options.get("task", DEFAULT_OPTIONS['task'])
        bboxes = options.get("bboxes", DEFAULT_OPTIONS['bboxes'])
        dialog = options.get("dialog", self.dialog)

        # 检测bbox和格式转换
        if bboxes is None:
            def detic_api(image: PILImage.Image, threshold: float) -> dict:
                import gradio_client
                if self.detic is None:
                    self.detic = gradio_client.Client(detic_url)
                options = json.dumps({"threshold": threshold})
                with tempfile.NamedTemporaryFile(suffix=".jpg") as image_tmp_file:
                    image.save(image_tmp_file)
                    response = self.detic.predict(image_tmp_file.name, options, api_name="/detic_api")
                return json.loads(response)
            bboxes = detic_api(image_color, detic_threshold)['bboxes']
            # print(f"detic_bboxes({len(bboxes)}):", bboxes)
        bboxes = np.asarray(bboxes).reshape(-1, 4)
        bboxes = json.dumps([{"bbox": b.tolist()} for b in bboxes])
        assert task in ["q", "g", "a"]
        assert isinstance(dialog, list)

        if task == "q":  # str: 问题
            text_response = self.func_ask(image=image_color, dialog=dialog, text=text, bboxes=bboxes)
            if text_response == "":
                bboxes, _ = self.func_guess(image=image_color, dialog=dialog, text=text, bboxes=bboxes)
            else:
                bboxes = None
            dialog += [[text, text_response]]
            done = len(text_response) == 0
        elif task == "g":  # list: 归一化的bbox
            text_response = ""
            bboxes, _ = self.func_guess(image=image_color, dialog=dialog, text="", bboxes=bboxes)
            done = False
        elif task == "a":  # str: 回答(oracle)
            text_response = self.func_answer(image=image_color, dialog=dialog, text=text, bboxes=bboxes)
            if len(dialog):  # 有历史对话
                dialog[-1][-1] = text
            else:  # 没有历史对话
                dialog += [[None, text]]
            dialog += [[text_response, None]]
            bboxes = None
            done = False
        else:
            raise RuntimeError(f"Not except error - task: {task}")
        self.dialog = dialog
        print(self.dialog)

        return {
            "text_response": text_response,
            "bboxes": bboxes,
            "done": done
        }


DEFAULT_OPTIONS = {
    "detic_url": "http://10.128.98.140:7863/",
    "detic_threshold": 0.5,
    "task": "q",
    "bboxes": None
}


if __name__ == "__main__":
    def main(config_file, port: int = None):
        from AppBase import main as gr_main
        return gr_main(model=ChatXVLM(), default_options=DEFAULT_OPTIONS, title="XVLM / Vilbert Demo", port=port)
    import fire
    fire.Fire(main)
