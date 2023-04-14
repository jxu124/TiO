import json
import re
import os
import torch
import jsonlines
import numpy as np
from tqdm import tqdm


PATH_TO_OPENIMAGES = "/mnt/bn/hri-lq/datasets/openimages_v1.2"
PATH_TO_COCO = "/mnt/bn/hri-lq/datasets/coco"
PATH_TO_INVIGTS = "/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns/dialog_translation.json"
PATH_TO_VISDIAL = "/mnt/bn/hri-lq/datasets/VisDial"
SEP_Q = "<sep_q>"
SEP_A = "<sep_a>"

USE_TQDM = True


xyxy_to_xywh = lambda bbox: bbox - np.concatenate([np.zeros_like(bbox[..., :2]), bbox[..., :2]], axis=-1)
xywh_to_xyxy = lambda bbox: bbox + np.concatenate([np.zeros_like(bbox[..., :2]), bbox[..., :2]], axis=-1)


def parse_dialog_en(dialog):
    dialog = re.split(f'{SEP_Q}|{SEP_A}', dialog)[1:]
    dialog = [s.strip() for s in dialog]
    ref_exp = dialog[0]
    q = []
    a = []
    for i, s in enumerate(dialog[1:]):
        if i % 2 == 0:
            q.append(s)
        else:
            a.append(s)
    qas = zip(q, a)
    qas = [{'q': q.strip(), 'a': a.strip()} for q, a in qas]
    return ref_exp, qas


# sbbox to bbox
def sbbox_to_bbox(sbbox, w, h, num_bins=1000):
    res = re.match("[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>[\ ]*<bin_([0-9]*)>(.*)", sbbox)
    if res is None:
        return None
    bbox = np.asarray([int(res.group(1))/num_bins, int(res.group(2))/num_bins, int(res.group(3))/num_bins, int(res.group(4))/num_bins])
    bbox = bbox * np.array([w, h, w, h])
    return bbox

    
def bbox_to_sbbox(bbox, w, h, num_bins=1000):
    bbox = np.asarray(bbox) / np.asarray([w, h, w, h])
    quant_x0 = "<bin_{}>".format(int((bbox[0] * (num_bins - 1)).round()))
    quant_y0 = "<bin_{}>".format(int((bbox[1] * (num_bins - 1)).round()))
    quant_x1 = "<bin_{}>".format(int((bbox[2] * (num_bins - 1)).round()))
    quant_y1 = "<bin_{}>".format(int((bbox[3] * (num_bins - 1)).round()))
    region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
    return region_coord


# ==== torch dataset ====
# InvigDataset
class InvigDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, 
                 path_dialog_ts=PATH_TO_INVIGTS, 
                 use_q=True,
                 use_a=True,
                 use_grounding=True):
        def build_image_path_map(path_to_openimages=PATH_TO_OPENIMAGES):
            """ invig中对openimages图片以hash保存，这个函数将建立一个map实现 hash->img_path的映射。输入需要openimages(v1.2)数据集的路径 """
            # 去除文件夹路径结尾的“/”
            path_to_openimages = f"{path_to_openimages.rstrip('/')}/imgs"
            # 载入图片名列表（无序）
            img_lst = os.listdir(path_to_openimages)
            # 使用正则表达式匹配hash，并生成映射
            img_hash_map = {
                re.match("([0-9a-f]*)_.*", i).group(1): f"imgs/{i}" for i in img_lst
            }
            return img_hash_map
        
        # 载入图片地址映射、载入翻译文件
        img_hash_map = build_image_path_map()
        with open(path_dialog_ts, "r") as f:
            ts = json.load(f)
            
        # 载入数据集标注文件
        # Invig 数据格式(迭代型)：data = [{
        # 'user_id': 'zhangyue0027', 
        # 'agent_id': 'zhengmenghan0012', 
        # 'label_id': 0, 
        # 'image_id': '06c5b78899bdb20f', 
        # 'dialog': 'zhangyue0027:哇，这些蛋糕好可爱，我想买一个。\nzhengmenghan0012:你想要哪一个？...', 
        # 'user_tgt': [6], 
        # 'agent_tgt': [6], 
        # 'source': 'v1', 
        # 'user_score': 1, 
        # 'user_familiarity': 1, 
        # 'agent_familiarity': 1, 
        # 'is_complete': True, 
        # 'objects': [
        #   {'bbox': [x, y, x, y], 'label': 'Muffin', 'segs': [], 'seg_labels': [], 'from_seg': False, 'label_id': 155, 'seg_label_ids': []}, 
        #   ...,
        #   {'bbox': [x, y, x, y], 'label': 'Dairy Product', 'segs': ['***.png'], 'seg_labels': ['Muffin'], 'from_seg': False, 'label_id': 357, 'seg_label_ids': [155]}, 
        #   ...]
        # }, ...]
        with open(file_path, "r") as f:
            data = json.load(f)
        data = tqdm(data, disable=not USE_TQDM)

        # 生成self.dataset
        self.dataset = []
        for _, d in enumerate(data):
            label_id = d['label_id']
            # user_id = d['user_id']
            # agent_id = d['agent_id']
            dialog = d['dialog'].split("\n")

            # a. 筛除偶数对话轮数（不可用数据）
            if len(dialog) % 2 != 1:
                continue
            for ii in range(len(dialog)):
                prefix = ["User:", "Robot:"]
                res = re.match('[a-z0-9\ ]*?:(.*)', dialog[ii])
                # b. 排除几个失败的样例
                if res is None:
                    continue
                dialog[ii] = f"{prefix[ii%2]}{res.group(1)}"
            dialog = '\n'.join(dialog)

            # c. 筛除没有翻译的对话
            idx = f'{label_id}'
            if idx not in ts:
                continue

            # d. 筛除不匹配对话
            if dialog != ts[idx]['before']:
                continue
            dialog = ts[idx]['after'].replace("Robot:", SEP_Q).replace("User:", SEP_A)

            # e. 删除未标注的样本
            if len(d['user_tgt']) < 1 :
                continue

            # f. 删除没有分类标签的样本
            j = d['user_tgt'][0]
            if 'seg_labels' in d['objects'][j]:
                obj_cate = d['objects'][j]['seg_labels'][0]
            elif 'label' in d['objects'][j]:
                obj_cate = d['objects'][j]['label']
            else:
                continue
                
            # dialog 分解为数据集需要的各项
            dialogue_id = f"invig.{label_id}"
            history, qas = parse_dialog_en(dialog)
            path_to_img = img_hash_map[d['image_id']]
            assert os.path.exists(f"{PATH_TO_OPENIMAGES}/{path_to_img}")
            region_coord = np.asarray(d['objects'][j]['bbox'])  # xyxy
            obj_cate = obj_cate.lower()

            item = {"id": dialogue_id, "img": path_to_img, "prompt": history, "qas": qas, "bbox": np.round(region_coord, 2).tolist(), "cate": obj_cate}
            # assert re.match(".*([\n]).*", json.dumps(item)) is None, json.dumps(item)
            self.dataset.append(item)
            # for i, qa in enumerate(qas):
            #     history = f"{history} {SEP_Q}"
            #     if use_q:
            #         uniq_id = f"invig.{dialogue_id}.{i:02d}.q"
            #         item = (uniq_id, path_to_img, history, qa['q'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['q']} {SEP_A}"
            #     if use_a:
            #         uniq_id = f"invig.{dialogue_id}.{i:02d}.a"
            #         item = (uniq_id, path_to_img, history, qa['a'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['a']}"
            # # history = f"{history}"
            # if use_grounding:
            #     uniq_id = f"invig.{dialogue_id}.grounding"
            #     item = (uniq_id, path_to_img, history, None, region_coord, obj_cate)
            #     self.dataset.append(item)
                
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


# GuesswhatDataset
class GuesswhatDataset(torch.utils.data.Dataset):
    def __init__(self, file_path,
                use_q=True,
                use_a=True,
                use_grounding=True):
        def get_coco_image(filename, path_to_coco=PATH_TO_COCO):
            if filename.startswith("COCO_val2014"):
                fullpath = f"val2014/{filename}"
            elif filename.startswith("COCO_train2014"):
                fullpath = f"train2014/{filename}"
            assert os.path.exists(f"{path_to_coco}/{fullpath}")
            return fullpath
        
        # 载入数据集标注文件
        # GuessWhat 数据格式(迭代型)：data = [{
        #   'status': 'success', 
        #   'picture': {'file_name': 'COCO_val2014_000000489209.jpg', 'flickr_url': ..., 'width': 480, 'coco_url': ..., 'height': 640}, 
        #   'picture_id': 489209, 
        #   'qas': [{'q': 'is it an bear?', 'a': 'Yes', 'id': 4970}, ...], 
        #   'questioner_id': 2, 
        #   'timestamp': '2016-07-08 15:06:58', 
        #   'object_id': 587892, 
        #   'dialogue_id': 2418, 
        #   'objects': {
        #       '587772': {'category': 'bear', 'area': 35610.1202, 'iscrowd': False, 'object_id': 587772, 'bbox': [x, y, w, h], 'category_id': 23, 'segment': [...]},
        #        ...
        #   }
        # }, ...]
        data = jsonlines.open(file_path)
        data = tqdm(data, disable=not USE_TQDM)
        
        # 生成self.dataset
        self.dataset = []
        for _, d in enumerate(data):
            # dialog 分解为数据集需要的各项
            dialogue_id = f"guesswhat.{d['dialogue_id']}"
            history = ""
            qas = d['qas']
            qas = [{"q": i["q"].capitalize(), "a": f"{i['a'].capitalize()}."} for i in qas]
            path_to_img = get_coco_image(d['picture']['file_name'])
            region_coord = xywh_to_xyxy(np.asarray(d['objects'][f"{d['object_id']}"]['bbox']))
            obj_cate = d['objects'][f"{d['object_id']}"]['category'].lower()
            
            # item = (dialogue_id, path_to_img, history, qas, region_coord, obj_cate)
            item = {"id": dialogue_id, "img": path_to_img, "prompt": history, "qas": qas, "bbox": np.round(region_coord, 2).tolist(), "cate": obj_cate}
            # assert re.match(".*([\n]).*", json.dumps(item)) is None, json.dumps(item)
            self.dataset.append(item)
            # for i, qa in enumerate(qas):
            #     history = f"{history} {SEP_Q}"
            #     if use_q:
            #         uniq_id = f"gw.{dialogue_id}.{i:02d}.q"
            #         item = (uniq_id, path_to_img, history, qa['q'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['q']} {SEP_A}"
            #     if use_a:
            #         uniq_id = f"gw.{dialogue_id}.{i:02d}.a"
            #         item = (uniq_id, path_to_img, history, qa['a'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['a']}"
            # # history = f"{history}"
            # if use_grounding:
            #     uniq_id = f"gw.{dialogue_id}.grounding"
            #     item = (uniq_id, path_to_img, history, None, region_coord, obj_cate)
            #     self.dataset.append(item)
        data.close()
                
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


# VisdialDataset
class VisdialDataset(torch.utils.data.Dataset):
    def __init__(self, file_path,
                use_q=True,
                use_a=True,
                use_grounding=None):
        def get_image_path(image_id, split="val2018", path_to_coco=PATH_TO_COCO, path_to_visdial=PATH_TO_VISDIAL):
            if split == "train":
                fullpath = f"train2014/COCO_train2014_{image_id:012d}.jpg"
                if not os.path.exists(f"{path_to_coco}/{fullpath}"):
                    fullpath = fullpath.replace("train2014", "val2014")
            elif split == "val2018":
                fullpath = f"val/VisualDialog_val2018/VisualDialog_val2018_{image_id:012d}.jpg"
            elif split == "test2018":
                fullpath = f"test/VisualDialog_test2018/VisualDialog_test2018_{image_id:012d}.jpg"
            assert os.path.exists(f"{path_to_coco}/{fullpath}") or f"{path_to_visdial}/{fullpath}"
            return fullpath
        
        # 载入数据集标注文件
        # VisDial 数据格式：data = {'version': '1.0', 'split': 'test2018', 'data': {
        #   'answers': ['a mini replica', ...], 
        #   'questions': ['does the shower have a glass door', ...],
        #   'dialogs': [
        #      {'image_id': 568676,
        #       'dialog': [{'answer': 4754, 'question': 12426}, ..., {'question': 18784, 'answer_options': [17773, ...]}],
        #       'caption': 'a woman is standing in front of a traffic light'},
        #       ...
        #   ]}
        with open(file_path) as f:
            data = json.load(f)
        data['data']['dialogs'] = tqdm(data['data']['dialogs'], disable=not USE_TQDM)
        qlist = data['data']['questions']
        alist = data['data']['answers']
        
        # 生成self.dataset
        self.dataset = []
        for _id, d in enumerate(data['data']['dialogs']):
            # dialog 分解为数据集需要的各项
            dialogue_id = f"visdial.{_id}"
            history = f'There is an image described by the text \"{d["caption"]}\".'
            qas = [{'q': f"{qlist[i['question']].capitalize()}?", 'a': f"{alist[i['answer']].capitalize()}."} for i in d['dialog'][:-1]]
            path_to_img = get_image_path(d['image_id'], data['split'])
            region_coord = []
            obj_cate = ""
            
            # item = (dialogue_id, path_to_img, history, qas, region_coord, obj_cate)
            item = {"id": dialogue_id, "img": path_to_img, "prompt": history, "qas": qas, "bbox": [], "cate": ""}
            # assert re.match(".*([\n]).*", json.dumps(item)) is None, json.dumps(item)
            self.dataset.append(item)
            # for i, qa in enumerate(qas):
            #     history = f"{history} {SEP_Q}"
            #     if use_q:
            #         uniq_id = f"vd.{dialogue_id}.{i:02d}.q"
            #         item = (uniq_id, path_to_img, history, qa['q'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['q']} {SEP_A}"
            #     if use_a:
            #         uniq_id = f"vd.{dialogue_id}.{i:02d}.a"
            #         item = (uniq_id, path_to_img, history, qa['a'], region_coord, obj_cate)
            #         self.dataset.append(item)
            #     history = f"{history} {qa['a']}"
                
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    process_items = [
        # invig
        # (InvigDataset, '/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns/train_v0.1.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/invig_train.jsonl'),
        # (InvigDataset, '/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns/valid_v0.1.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/invig_valid.jsonl'),
        # (InvigDataset, '/mnt/bn/hri-lq/datasets/invig/invig_v0.1_anns/anns/test_v0.1.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/invig_test.jsonl'),

        # # Guesswhat
        # (GuesswhatDataset, '/mnt/bn/hri-lq/datasets/guesswhat/guesswhat.train.jsonl', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/guesswhat_train.jsonl'),
        # (GuesswhatDataset, '/mnt/bn/hri-lq/datasets/guesswhat/guesswhat.valid.jsonl', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/guesswhat_valid.jsonl'),
        # (GuesswhatDataset, '/mnt/bn/hri-lq/datasets/guesswhat/guesswhat.test.jsonl', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/guesswhat_test.jsonl'),

        # # Visdial
        (VisdialDataset, '/mnt/bn/hri-lq/datasets/VisDial/train/visdial_1.0_train.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/visdial_train.jsonl'),
        # (VisdialDataset, '/mnt/bn/hri-lq/datasets/VisDial/val/visdial_1.0_val.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/visdial_valid.jsonl'),
        # (VisdialDataset, '/mnt/bn/hri-lq/datasets/VisDial/test/visdial_1.0_test.json', '/mnt/bn/hri-lq/datasets/ofa/dialog_data/visdial_test.jsonl'),

    ]
    # main loop
    for cls, load_path, save_path in process_items:
        print('*'*20)
        print(f"Convert {load_path} to {save_path}")
        ds = cls(load_path)
        with jsonlines.open(save_path, mode='w') as writer:
            for i in tqdm(ds.dataset, postfix="Saving...", disable=not USE_TQDM):
                writer.write(i)
