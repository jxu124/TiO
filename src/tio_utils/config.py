# Author: Jie Xu
# Date: 4/2/2023

# builtin library
from typing import Optional, Any, Union, List
import json
import argparse

# 3rd library
from PIL import Image
import yaml
import numpy as np

# self library
from .data import get_processor


import logging
logger = logging.getLogger(__name__)


class TiOConfig():
    def __init__(self, path_config) -> None:
        with open(path_config) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(self.cfg['env']['path_exclude_images'], "r") as f:
            self.test_images = set(json.load(f))
        self.path_images = self.cfg['env']['path_images']
        self.tokenizer, self.image_processor = self.get_processor()

    def get_processor(self):
        return get_processor(pretrain_path=self.cfg['env']['path_tokenizer'])

    def filter_exclude_test_images(self, global_image_id):
        # usage: ds.filter(tio_config.filter_exclude_test_images, input_columns=['global_image_id'])
        if type(global_image_id) is list:
            return [(i not in self.test_images) for i in global_image_id]
        elif type(global_image_id) is str:  # str
            return global_image_id not in self.test_images
        else:
            raise ValueError

    def load_image(self, features):
        # usage: ds.map(MapFunc.load_images, input_columns=['image_path'])
        image_path = features['image_path']
        for k, v in self.path_images.items():
            image_path = image_path.replace(k, v, 1)
        return {**features, 'image': Image.open(image_path)}

    def get_dataset(self, split: str):
        import datasets
        # 1 按照node切分数据集
        datasets_collection = []
        for i in self.cfg[split]:
            if i.get('streaming'):
                continue
            name = i['name']
            logger.info(f"loading {name}-{split} from {i['path']}")
            ds = datasets.load_from_disk(i['path'])[split]
            for task, prob in zip(i['tasks'], i['probs']):
                ds_task = ds.add_column("__task__", [task] * len(ds))\
                            .filter(self.filter_exclude_test_images, input_columns=['global_image_id'])
                datasets_collection += [(name, task, ds_task, prob)]

        # 2 按照比例组合数据
        tasks = [i[1] for i in datasets_collection]
        use_datasets = [i[2] for i in datasets_collection]
        probs = np.asarray([i[3] for i in datasets_collection])
        counts = np.asarray([len(ds) for ds in use_datasets])
        n_samples = (probs * counts).astype(int)
        total_count = sum(n_samples) // 512 * 512
        n_samples[-1] = total_count - sum(n_samples[:-1])
        use_datasets = [ds.shuffle().select(range(n)) for ds, n in zip(use_datasets, n_samples)]

        # 3 构造fairseq_dataset，送进去预处理函数
        dataset = datasets.concatenate_datasets(use_datasets)

        # 4 logging
        logger.info(f"load {split} data ({len(dataset)}/{total_count} samples): {len(use_datasets)} dataset(s)")
        logger.info("Tasks: " + ", ".join([f'{t}({n})' for t, n in zip(tasks, n_samples)]))
        return dataset


# import torchvision.transforms as T
# def get_image_processor(resolution=512, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
#     image_processor = T.Compose([
#         lambda image: image.convert("RGB"),
#         T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=mean, std=std)
#     ])
#     return image_processor
