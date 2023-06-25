# Author: ***
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
from .. import path_tokenizer


import logging
logger = logging.getLogger(__name__)


class TiOConfig():
    _cfg = None
    _test_images = None
    _tokenizer = None
    _image_processor = None

    def __init__(self, path_config) -> None:
        self.path_config = path_config

    @property
    def cfg(self):
        if self._cfg is None:
            with open(self.path_config) as f:
                self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        return self._cfg

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.get_processor()
        return self._tokenizer

    @property
    def image_processor(self):
        if self._image_processor is None:
            self.get_processor()
        return self._image_processor

    @property
    def test_images(self):
        if self._test_images is None:
            with open(self.cfg['env']['path_exclude_images'], "r") as f:
                self._test_images = set(json.load(f))
        return self._test_images

    def get_processor(self):
        self._tokenizer, self._image_processor = get_processor(pretrain_path=path_tokenizer)
        return self.tokenizer, self.image_processor

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
        for k, v in self.cfg['env']['path_images'].items():
            image_path = image_path.replace(k, v, 1)
        return {**features, 'image': Image.open(image_path)}

    def get_dataset(self, split: str):
        import datasets
        # 1 load datasets and set tasks
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

        # 2 weighted datasets
        tasks = [i[1] for i in datasets_collection]
        use_datasets = [i[2] for i in datasets_collection]
        probs = np.asarray([i[3] for i in datasets_collection])
        counts = np.asarray([len(ds) for ds in use_datasets])
        n_samples = (probs * counts).astype(int)
        total_count = sum(n_samples) // 512 * 512
        n_samples[-1] = total_count - sum(n_samples[:-1])
        use_datasets = [ds.shuffle().select(range(n)) for ds, n in zip(use_datasets, n_samples)]

        # 3 concat datasets
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
