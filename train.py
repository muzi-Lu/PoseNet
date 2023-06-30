import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataloader
from models.models import create_model
from util.visualizer import Visualizer
from util.common.config_parser import AbsPoseConfig
from util.common.setup import *

import torch
import numpy
import random

def setup_config(config):
    print('Setup configurations...')
    # Seedings
    make_deterministic(config.seed)

    # Setup logging dir

    # Setup running devices

    # Setup datasets

    # Define image preprocessing

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



if __name__ == '__main__':
    main()
