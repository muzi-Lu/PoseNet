import os.path
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataloader
from models.models import create_model
from util.visualizer import Visualizer
from util.common.config_parser import AbsPoseConfig
from util.common.setup import *
from util.datasets.abspose import AbsPoseDataset
from util.datasets.preprocess import *

import torch
import numpy
import random


def setup_config(config):
    print('Setup configurations...')
    print(config)
    # Seedings
    make_deterministic(config.seed)

    # Setup logging dir
    if not os.path.exists(config.odir):
        os.makedirs(config.odir)
    config.log = os.path.join(config.odir, 'log_txt') if config.training else os.path.join(config.odir, 'test_result.txt')
    config.ckpt_dir = os.path.join(config.odir, 'ckpt')
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    print(config)

    # Setup running devices
    if torch.cuda.is_available():
        print('Use GPU device:{}.'.format(config.gpu))
        config.device = torch.device('cuda:{}'.format(config.gpu))
    else:
        print('No GPU available, use CPU device.')
        config.device = torch.device("cpu")
    delattr(config, 'gpu')
    print(config)

    # Setup datasets
    config.data_class = AbsPoseDataset

    # Define image preprocessing
    im_mean = os.path.join(config.data_root, config.dataset, config.image_mean) if config.image_mean else None
    if config.crop:
        crop = 'random' if config.training else 'center'
    else:
        crop = None
    config.ops = get_transform_ops()
    config.val_ops = get_transform_ops()

    # Model initialization

    # Setup optimizer

def train():
    pass

def test():
    pass

def main():
    # 主函数了，这个时候首先是全部的参数传递
    config = AbsPoseConfig().parse()

    setup_config(config)

    log = open(config.log, 'a')



if __name__ == '__main__':
    main()
