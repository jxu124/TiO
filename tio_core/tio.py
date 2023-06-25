#!/usr/bin/python3

from __future__ import annotations

from PIL import Image as PILImage
import numpy as np

from tio_core import path_ckpt
from tio_core.utils import sbbox_to_bbox
from tio_core.module import OFAModelWarper


class ChatTiO():
    def __init__(self, path_ckpt=path_ckpt):
        self.model = OFAModelWarper(path_ckpt)
        print('Initialization Finished')

    def reset(self, rgb: PILImage.Image = None, config: dict = {}):
        self.image = rgb

        # 场景感知
        self.dialog = []
        self.raw_dialog_log = {"dialog": self.dialog}
        self.imgs = {"rgb": rgb}
        self.done = False

    def chat(self, text: str) -> str:
        assert not self.done
        def get_bboxes(message):
            if "region:" in message:
                bboxes = sbbox_to_bbox(message) * np.asarray([*self.image.size, *self.image.size])
                bboxes = [b.astype(int).reshape(4).tolist() for b in bboxes]
            else:
                bboxes = []
            return bboxes

        # 获取对话结果
        text_dialog = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in self.dialog])
        text_dialog += f" human: {text}\n"
        text_response = self.model.generate([text_dialog.lower()], [self.image])[0]
        bboxes_1 = get_bboxes(text_response)

        # 获取grounding结果
        text_dialog += f" agent: {text_response}\n"
        text_grounding_query = f" \n#instruction: which region does the context describe?\n#context: \"{text_dialog}\""
        text_grounding = self.model.generate([text_grounding_query.lower()], [self.image])[0]
        bboxes_2 = get_bboxes(text_grounding)
        print(text_grounding_query)

        self.dialog += [(text, text_response)]
        others = {"bboxes": bboxes_2 + bboxes_1}
        return text_response, others
