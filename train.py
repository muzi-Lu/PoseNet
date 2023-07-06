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

import torch.utils.data as data
import networks

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

    if config.validate:
        config.validate = config.validate[0]

    # Setup datasets
    config.data_class = AbsPoseDataset

    # Define image preprocessing
    im_mean = os.path.join(config.data_root, config.dataset, config.image_mean) if config.image_mean else None
    if config.crop:
        crop = 'random' if config.training else 'center'
    else:
        crop = None
    config.ops = get_transform_ops(config.rescale, im_mean, crop, crop_size=config.crop, normalize=config.normalize)
    config.val_ops = get_transform_ops(config.rescale, im_mean, 'center', crop_size=config.crop, normalize=config.normalize)
    delattr(config, 'crop')
    delattr(config, 'rescale')
    delattr(config, 'normalize')

    # Model initialization
    config.start_epoch = 0
    config.weights_dir = None
    config.weights_dict = None
    config.optimizer_dict = None
    if config.pretrained:
        config.weights_dir = config.pretrained[0]
        config.weights_dict = torch.load(config.weights_dir)
    if config.resume:
        config.weights_dir = config.resume[0]
        checkpoint = torch.load(config.weights_dir)
        assert config.network == checkpoint['network']
        config.start_epoch = checkpoint['last_epoch'] + 1
        config.weights_dict = checkpoint['state_dict']
        config.optimizer_dict = checkpoint['optimizer']
    delattr(config, 'resume')
    delattr(config, 'pretrained')

    # Setup optimizer
    optim = config.optim
    optim_tag = ''
    if config.optim == 'Adam':
        optim_tag = 'Adam_eps{}'.format(config.epsilon)
        delattr(config, 'momentum')
    elif config.optim == 'SGD':
        optim_tag = 'SGD_mom{}'.format(config.mementum)
        delattr(config.epsilon)
    optim_tag = '{}_{}'.format(optim_tag, config.lr_init)

    if config.lr_decay:
        config.lr_decay_step = int(config.lr_decay[1])
        config.lr_decay_factor = int(config.lr_decay[0])
        config.lr_decay = True
        optim_tag = '{}_lrd{}-{}'.format(optim_tag, config.lr_decay_step, config.lr_decay_factor)
    optim_tag = '{}_wd{}'.format(optim_tag, config.weight_decay)
    config.optim_tag = optim_tag

def train(net, config, log, train_loader, val_loader=None):
    optim_search = True
    # Setup visualizer
    if not optim_search:
        pass
    else:
        pass


def train():
    pass

def test():
    pass

def main():
    # 主函数了，这个时候首先是全部的参数传递
    config = AbsPoseConfig().parse()

    setup_config(config)
    log = open(config.log, 'a')
    lprint(config_to_string(config), log)
    print(config)

    # Datasets configuration
    data_src = AbsPoseDataset(config.dataset, config.data_root, config.val_pose_txt, config.ops)
    data_loader = data.DataLoader(data_src, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    lprint('Dataset total samples: {}'.format(len(data_src)))

    if config.validate:
        val_data_src = AbsPoseDataset(config.dataset, config.data_root, config.val_pose_txt, config.val_ops)
        val_looader = data.DataLoader(val_data_src, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    else:
        val_loader = None

    if config.weig

    if config.validate:
        pass



if __name__ == '__main__':
    main()
