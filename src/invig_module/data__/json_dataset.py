# Author: Jie Xu
# Date: 4/2/2023

import os
import json
import torch

from ofa_module import data as ofa_module_data
FileDataset = ofa_module_data.file_dataset.FileDataset


class JsonlDataset(FileDataset):
    def __init__(self, file_path, cached_index=False):
        self.file_path = file_path
        assert os.path.exists(self.file_path), "Error: The local datafile {} not exists!".format(self.file_path)

        self.data_cnt = 0
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1
        self.cached_index = cached_index
        self._init_seek_index()
        self._reader = self._get_reader()
        print("file {} slice_id {} row count {} total row count {}".format(
            self.file_path, self.slice_id, self.row_count, self.total_row_count)
        )

    def __getitem__(self, index):
        if self.data_cnt == self.row_count:
            print("reach the end of datafile, start a new reader")
            self.data_cnt = 0
            self._reader = self._get_reader()
        self.data_cnt += 1
        column_l = json.loads(self._reader.readline())
        # column_l = self._reader.readline().rstrip("\n").split(self.separator)
        # column_l = [dtype(column_l[col_id]) for col_id, dtype in zip(self.selected_col_ids, self.dtypes)]
        return column_l
