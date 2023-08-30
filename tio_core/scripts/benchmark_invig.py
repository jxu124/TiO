# Author: Xu Jie
# Date: 06/19/23

# builtin-lib
from typing import List, Tuple
import os
import sys
import time
import argparse
import json
import datetime
import pickle
import pathlib
import tempfile
import datasets

# 3rd-lib
from tqdm.auto import tqdm
from PIL import Image as PILImage
import numpy as np
import gradio_client


BASE_PATH = pathlib.Path(os.path.abspath(__file__)).parent


# ===========================================================
# ○ module define
# ===========================================================
def iou(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        box_a[:, 2:][:, None, :].repeat(B, axis=1),
        box_b[:, 2:][None, :, :].repeat(A, axis=0)
    )
    min_xy = np.maximum(
        box_a[:, :2][:, None, :].repeat(B, axis=1),
        box_b[:, :2][None, :, :].repeat(A, axis=0)
    )
    inter_size = np.maximum((max_xy - min_xy), 0)
    inter = inter_size[:, :, 0] * inter_size[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / union


def eval_grounding_acc(
        grounding_results: np.ndarray,  # shape(*, 4)
        grounding_annotations: np.ndarray,  # shape(*, 4)
        with_gt_bbox=True,
        bin_cls_thresh=0.5,
):
    iou_thresh = np.arange(0.1, 1.0, 0.1)
    iou_thresh = np.append(iou_thresh, 0.99)
    total = 0
    tp = np.zeros(iou_thresh.shape)
    grounding_results = np.asarray(grounding_results).reshape([-1, 4])
    grounding_annotations = np.asarray(grounding_annotations).reshape([-1, 4])
    for res, ann in zip(grounding_results, grounding_annotations):
        # pred_bbox = res['pred_ref_bboxes'][-1][None, :]
        # gt_bbox = ann['ref_bboxes']
        pred_bbox = res[None, ...]
        gt_bbox = ann[None, ...]
        _iou = iou(pred_bbox, gt_bbox)
        total += 1
        tp += (iou_thresh < _iou.max())

    results = dict(zip(
        [f'ap_{i * 10}' for i in range(1, 10)],
        np.round(tp[:-1] / total, 4)
    ))
    results['ap_99'] = np.round(tp[-1] / total, 4)

    # TODO: Use object bounding boxes to check the predictions
    if with_gt_bbox:
        bin_pos_correct = 0
        bin_neg_correct = 0
        bin_pos_total = 0
        bin_neg_total = 0

        if 'pred_grounding_logits' not in grounding_results[0]:
            import warnings
            warnings.warn("No predicted logits were found in the results. "
                          "The evaluation on the binary grounding accuracy "
                          "will be skipped. REMEMBER to set with_gt_proposals "
                          "to False if you use bbox regression for visual grounding.")
            return results

        for res, ann in zip(grounding_results, grounding_annotations):
            gt_ids = iou(ann['ref_bboxes'], ann['bboxes']).argmax(-1)
            _pos_masks = np.zeros(ann['bboxes'].shape[0], dtype=np.bool)
            _pos_masks[gt_ids] = True
            _neg_masks = ~_pos_masks
            pred_logits = res['pred_grounding_logits'][-1]
            num_obj = len(pred_logits)
            bin_pos_correct += (pred_logits[_pos_masks] > bin_cls_thresh).sum()
            bin_pos_total += gt_ids.size
            bin_neg_correct += (pred_logits[_neg_masks] <= bin_cls_thresh).sum()
            bin_neg_total += num_obj - gt_ids.size

        results.update(dict(
            binary_pos_acc=np.round(float(bin_pos_correct) / float(bin_pos_total), 4),
            binary_neg_acc=np.round(float(bin_neg_correct) / float(bin_neg_total), 4)
        ))

    return results


methods = ["TiO", "XVLM", "Seeask2"]
method_urls_default = [["http://10.128.100.214:7864"], ["http://10.128.96.82:7860"], ["http://10.128.98.140:7862"]]
method_options = {
    "method": "seeask",
    "config": "vocab+reclip",
    "detic_url": "http://10.128.98.140:7863",
    "detic_threshold": 0.5,
    "task": "g"
}


def main():
    def parse_item(item):
        image = item["image_raw"]
        bbox = item['bbox'] / np.asarray([*image.size, *image.size])
        dialog = sum(item["dialog"], [])[1:] + [""]
        dialog = [[dialog[i*2], dialog[i*2+1]] for i in range(len(dialog) // 2)]
        return image, dialog, bbox

    def make_context(dialog, text_input=None):
        if dialog[-1][1] in ["", None]:
            assert text_input is None
            text_input = dialog[-1][0]
            dialog = dialog[:-1]
        context = "".join([f" human: {i[0]}\n agent: {i[1]}\n" for i in dialog])
        return context if text_input is None else context + f" human: {text_input}"

    method_idx = 0
    method_str = methods[method_idx]
    method_client = gradio_client.Client(method_urls_default[method_idx][0])
    print(f"Method Select: {method_str}({method_urls_default[method_idx][0]})")

    bbox_gt = []
    bbox_pred = []
    ds = datasets.load_dataset("parquet", split="train", data_files="hdfs:///home/byte_ailab_roboticsresearch/user/xujie/hf_datasets/invig/test-000000-of-000001.parquet")
    for i in tqdm(ds):
        image, dialog, bbox = parse_item(i)
        text_input = make_context(dialog)
        method_client.predict(api_name="/reset")
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            image.save(f)
            result = method_client.predict(f.name, text_input, json.dumps(method_options), api_name="/chat")
            result = json.loads(result)
        bbox_gt += [bbox]
        bbox_pred += [result['bboxes'][0]]
    res = eval_grounding_acc(bbox_pred, bbox_gt, False)

    from terminaltables import AsciiTable
    table_data = [["ap", "score"]] + [[i, j] for i, j in res.items()]
    print(AsciiTable(table_data).table)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
