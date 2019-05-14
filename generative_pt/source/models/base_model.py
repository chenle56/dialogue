#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/base_model.py
"""

import os
import torch
import torch.nn as nn

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd(用于自动求导)之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)

    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd自动求导)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.

    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：

    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28

"""
class BaseModel(nn.Module):
    """
    BaseModel :__init__,forward,__repr__,save,load,
    """
    def __init__(self):
        super(BaseModel, self).__init__()      #网络中具有可学习参数的层

    def forward(self, *input):
        """
        forward
        """
        raise NotImplementedError

    def __repr__(self):       #查看对象
        main_string = super(BaseModel, self).__repr__()  #查看对象的时候调用
        num_parameters = sum([p.nelement() for p in self.parameters()]) # nelement() 统计 tensor (张量) 的元素的个数      .parameters()获取网络的参数
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string

    def save(self, filename):
        """
        save    Module.state_dict()Returns a dictionary containing a whole state of the module.
        """
        torch.save(self.state_dict(), filename)
        print("Saved model state to '{}'!".format(filename))

    def load(self, filename):
        """
        load
        """
        if os.path.isfile(filename):   # .isfile()   Test whether a path is a regular file
            state_dict = torch.load(    #Loads an object saved with :func:`torch.save` from a file
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)     #Loads an object saved with :func:`torch.save` from a file
            print("Loaded model state from '{}'".format(filename))
        else:
            print("Invalid model state file: '{}'".format(filename))
