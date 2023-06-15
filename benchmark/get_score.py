import numpy as np
import warnings
import json
import os


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
        grounding_results,
        grounding_annotations,
        with_gt_bbox=True,
        bin_cls_thresh=0.5,
):
    iou_thresh = np.arange(0.1, 1.0, 0.1)
    iou_thresh = np.append(iou_thresh, 0.99)
    total = 0
    tp = np.zeros(iou_thresh.shape)
    for res, ann in zip(grounding_results, grounding_annotations):
        pred_bbox = res['pred_ref_bboxes'][-1][None, :]
        gt_bbox = ann['ref_bboxes']
        _iou = iou(pred_bbox, gt_bbox)
        total += 1
        tp += (iou_thresh < _iou.max())

    results = dict(zip(
        [f'grounding_acc_iou_{float(i) / 10}' for i in range(1, 10)],
        np.round(tp[:-1] / total, 4)
    ))
    results['grounding_acc_iou_0.99'] = np.round(tp[-1] / total, 4)

    # TODO: Use object bounding boxes to check the predictions
    if with_gt_bbox:
        bin_pos_correct = 0
        bin_neg_correct = 0
        bin_pos_total = 0
        bin_neg_total = 0

        if 'pred_grounding_logits' not in grounding_results[0]:
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


def grounding_acc(grounding_json):
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../src"))
    from tio_utils import sbbox_to_bbox
    
    with open(grounding_json) as f:
        data = f.readlines()
    data_1 = [json.loads(i) for i in data[1:]]
    pred_ref_bboxes = [{'pred_ref_bboxes': np.asarray(sbbox_to_bbox(d[3])).reshape(-1, 4)} for d in data_1]
    all_target = [{'ref_bboxes': np.asarray(sbbox_to_bbox(d[2])).reshape(-1, 4)} for d in data_1]
    return eval_grounding_acc(pred_ref_bboxes, all_target)


def grounding_acc_xvlm(grounding_json):
    with open(grounding_json) as f:
        data = f.readlines()
    data_1 = [json.loads(i) for i in data[1:]]
    pred_ref_bboxes = [{'pred_ref_bboxes': np.asarray(json.loads(d[5])).reshape(-1, 4)} for d in data_1]
    all_target = [{'ref_bboxes': np.asarray(d[6]).reshape(-1, 4)} for d in data_1]
    return eval_grounding_acc(pred_ref_bboxes, all_target)


if __name__ == "__main__":
    # python3 ./benchmark/get_score.py --file ./xvlm_logs/invig_grounding/test/20230522-153744.jsonl
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    main_args = parser.parse_args()

    try:
        results = grounding_acc_xvlm(main_args.file)
    except:
        results = grounding_acc(main_args.file)
    print(results)
