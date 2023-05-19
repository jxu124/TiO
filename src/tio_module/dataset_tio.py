from fairseq.data import FairseqDataset
from datasets.distributed import split_dataset_by_node
import datasets
import argparse

from tio_utils import DataCollator, DataPrompt, TiOConfig, world_info_from_env


datasets.config.IN_MEMORY_MAX_SIZE = 1_024_000_000  # 1GB


class TiODataset(FairseqDataset):  # ~OFADataset
    def __init__(self, dataset: datasets.Dataset, tio_config: TiOConfig, distributed_split: bool = True, **args):
        _, rank, world_size = world_info_from_env()
        total_count = len(dataset)
        self.style = None
        self.ds = split_dataset_by_node(dataset, rank, world_size) if distributed_split else dataset
        self.dataset = argparse.Namespace(**{'get_total_row_count': lambda: total_count, "_seek": lambda offset=0: None})
        self.tio_config = tio_config
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = self.tio_config.tokenizer
        self.image_processor = self.tio_config.image_processor
        # self.tokenizer, self.image_processor = self.tio_config.get_processor()
        self.collater_args = args

    def __getitem__(self, idx):
        data = self.ds[idx]
        return getattr(DataPrompt, data['__task__'])(self.tio_config.load_image(data), self.style)

    def __len__(self):
        return len(self.ds)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(epoch)
        elif hasattr(self.ds, "shuffle"):
            self.ds.shuffle(epoch)

    def collater(self, samples, pad_to_length=None):
        collate_fn = DataCollator(tokenizer=self.tokenizer,
                                  image_processor=self.image_processor,
                                  max_src_length=self.tio_config.cfg['hparam']['max_src_length'],
                                  max_tgt_length=self.tio_config.cfg['hparam']['max_tgt_length'],
                                  **self.collater_args)  # or max_length
        return collate_fn(samples)
