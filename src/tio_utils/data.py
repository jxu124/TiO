# builtin library & 3rd library
import re
import numpy as np
import transformers

import logging
logger = logging.getLogger(__name__)


def sbbox_to_bbox(sbbox, w=None, h=None, num_bins=1000, strict=False):
    """
    Convert string bbox token (e.g. <bin_0> <bin_1> <bin_2> <bin_3>) to bboxes (shape: [-1, 4]).

    The sbbox_to_bbox function takes the following arguments:
    sbbox: The string representation of the sbbox.
    w: The width of the original image.
    h: The height of the original image.
    num_bins: The number of bins in the 1000x1000 grid.
    strict: A boolean flag indicating whether to raise an exception if the sbbox is not in the correct format.
    """
    # patten = re.compile(r"<bin_(\d+)> <bin_(\d+)> <bin_(\d+)> <bin_(\d+)>")
    # ret = [re.match(patten, item) for item in sbbox.split("region: ")]
    # ret = [[i.group(1), i.group(2), i.group(3), i.group(4)] for i in ret if i is not None]
    # bbox = np.array(ret, dtype=int) / num_bins
    bbox = np.asarray(re.findall(r"<bin_(\d+)>", sbbox), dtype=int)
    bbox = bbox[:len(bbox) // 4 * 4].reshape(-1, 4) / num_bins
    if not strict and bbox.size == 0:
        bbox = np.array([[0, 0, 1, 1]])
    bbox = np.clip(bbox, 1e-3, 1 - 1e-3)
    if w is not None and h is not None:
        bbox = bbox * np.array([w, h, w, h])
    return bbox.reshape(-1, 4)


def bbox_to_sbbox(bbox, w=None, h=None, num_bins=1000):
    """ This function converts a dense bounding box (bbox) to a string bounding box (sbbox). """
    if w is not None and h is not None:
        bbox = np.asarray(bbox).reshape(4) / np.asarray([w, h, w, h])
    bbox = np.clip(bbox, 1e-3, 1 - 1e-3)
    quant_x0 = "<bin_{}>".format(int((bbox[0] * (num_bins - 1)).round()))
    quant_y0 = "<bin_{}>".format(int((bbox[1] * (num_bins - 1)).round()))
    quant_x1 = "<bin_{}>".format(int((bbox[2] * (num_bins - 1)).round()))
    quant_y1 = "<bin_{}>".format(int((bbox[3] * (num_bins - 1)).round()))
    region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
    return region_coord


def get_processor(pretrain_path='jxu124/TiO', resolution=512):
    """ get_processor() -> return (tokenizer, image_processor) """
    from transformers_ofa.tokenization_ofa import OFATokenizer
    tokenizer = OFATokenizer.from_pretrained(pretrain_path, truncation_side="left")
    image_processor = transformers.AutoImageProcessor.from_pretrained(pretrain_path, size=resolution)
    return (tokenizer, image_processor)
