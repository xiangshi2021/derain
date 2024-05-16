# -*- coding: utf-8 -*-
"""
运行model

"""

from options.train_options import TrainOptions
from train_module import trainer

opt = TrainOptions().parse()
tr = trainer(opt)
tr.train_start()

