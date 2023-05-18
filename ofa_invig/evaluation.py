# copy from https://code.byted.org/robotics/alpha_vision/blob/dev/avision/core/grounding/evaluation.py
import warnings

import numpy as np
# from ..utils import iou
# from mmdet.models.losses import accuracy
# from nlgeval import compute_metrics
import pycocoevalcap
import torch


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

    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))[:, None]  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1]))[None, :]
    union = area_a + area_b - inter

    return inter / union


# pred_ref_bboxes, ref_bboxes
# pred_grounding_logits, (ref_bboxes, bboxes)
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


def _calculate_ap_score(hyps, refs, thresh=0.5):
    interacts = torch.cat(
        [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
         torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
        dim=1
    )
    area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
    area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
    interacts_w = interacts[:, 2] - interacts[:, 0]
    interacts_h = interacts[:, 3] - interacts[:, 1]
    area_interacts = interacts_w * interacts_h
    ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
    return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()


def eval_grounding_acc_v2(grounding_res, grounding_ann):
    """
    grounding_res: (N, 4)
    grounding_ann: (N, 4)
    """
    ap_scores = {}
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        ap_score = _calculate_ap_score(torch.tensor(grounding_res), torch.tensor(grounding_ann), t)
        ap_scores[f"grounding_acc_iou_{t}"] = ap_score.sum().item() / len(ap_score)
    return ap_scores


def eval_qgen_word_acc(
        qgen_results,  # [{'pred_question_tokens': ...}, ...]
        qgen_annotations,  # [{'questions': ...}, ...]
        consider_eos=True, # whether to consider eos when computing the accuracy of question generation
        eos_token_id=None
):
    # when computing the accuracy, the correct prediction of EOS will make results better
    if not consider_eos:
        assert eos_token_id is not None
    total_token_num = 0
    correct_token_num = 0
    for res, ann in zip(qgen_results, qgen_annotations):
        pred_tokens = res['pred_question_tokens']
        gt_tokens = ann['questions']
        for pred, gt in zip(pred_tokens, gt_tokens):
            gt = gt[gt > 0]
            gt = gt[1:][:len(pred)] # prevent long questions that exceeds the max length of the model outputs
            assert len(gt) == len(pred)
            if not consider_eos:
                _keep = gt != eos_token_id
            else:
                _keep = np.ones(gt.shape, dtype=np.bool)
            total_token_num += _keep.sum()
            correct_token_num += (gt[_keep] == pred[_keep]).sum()
    return np.round(float(correct_token_num) / float(total_token_num), 4)



if __name__ == "__main__":
    pass

    # pred_ref_bboxes, ref_bboxes
    grounding_results = [{"pred_ref_bboxes": np.array([[0, 0, 200, 400.]])}, {"pred_ref_bboxes": np.array([[0, 0, 200, 200.]])}]
    grounding_annotations = [{"ref_bboxes": np.array([[0, 0, 200, 200.]])}, {"ref_bboxes": np.array([[0, 0, 200, 200.]])}]
    res = eval_grounding_acc(grounding_results, grounding_annotations)

    print(res)

