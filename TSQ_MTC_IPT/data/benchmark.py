import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data


'''这段代码定义了一个用于名为 Benchmark 的数据集类，
专门用于处理基准测试（Benchmark）数据集，继承自超分辨率数据处理基类 srdata.SRData。
它的主要作用是为模型性能评估提供标准化的基准测试数据加载功能。'''
class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        # import pdb; pdb.set_trace()
        # self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.apath = os.path.join('benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')

