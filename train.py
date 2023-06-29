import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataloader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

import torch
import numpy
import random
torch.manual_seed(opt.seed)
numpy.random.seed(opt.seed)
random.seed(opt.seed)

# torch.backends.cudnn.deterministic = True
# 这个是要下Cudnn吗

data_loader = CreateDataloader(opt)
# dataset = data_loader.loa

def main():
    # 主函数了，这个时候首先是全部的参数传递
    config =

if __name__ = '__main__':
    main()