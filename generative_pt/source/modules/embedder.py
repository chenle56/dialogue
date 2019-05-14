#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/embedder.py
"""

import torch
import torch.nn as nn


class Embedder(nn.Embedding):
    """
    Embedder:load_embeddings
    nn.Embedding:A simple lookup table that stores embeddings of a fixed dictionary and size.  This module is often used to store word embeddings and retrieve them using indices.
    """
    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)  #一种包含单一数据类型元素的多维矩阵是  torch.FlaotTensor的简称
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                """
                nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数,只有非零元素才会有索引值，零值元素没有索引值
                transpose(np.nonzero(x))函数能够描述出每一个非零元素在不同维度的索引值
                通过a[nonzero(a)]得到所有a中的非零值
                """
                nn.init.uniform_(embeds[i], -scale, scale) #从均匀分布U(a, b)中生成值，填充输入的张量或变量
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))
