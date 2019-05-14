#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/dataset.py
"""

import torch
from torch.utils.data import DataLoader

from source.utils.misc import Pack
from source.utils.misc import list2tensor


class Dataset(torch.utils.data.Dataset):
    """
    Dataset:
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):   # 提供数据集的大小 ,此处即为sample.train.txt的大小（拆成单轮会话的大小）
        return len(self.data)

    def __getitem__(self, idx):   # 支持整数索引（索引为第i-1轮单轮会话），范围从0到len(self)
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
        数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader
