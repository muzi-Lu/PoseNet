import torch
import random
import numpy as np
from collections import OrderedDict

def load_weights_to_gpu(weights_dir=None, gpu=None):
    weights_dict = None
    if weights_dir is not None:
        if gpu is not None:
            map_location = lambda storage, loc: storage.cuda(gpu)
        else:
            map_location = lambda storage, loc: storage
        weights = torch.load(weights_dir, map_location=map_location)
        if isinstance(weights, OrderedDict):
            weights_dict = weights
        elif isinstance(weights, dict) and 'state_dict' in weights:
            weights_dict = weights['state_dict']
    return weights_dict

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # 没有
    torch.backends.cudnn.benchmark = False # 没有阿

def lprint(ms, log=None):
    '''
    Print message to console and to a log file
    :param ms:
    :param log:
    :return:
    '''
    print(ms)
    if log:
        log.write(ms+'/n')
        log.flush()

def config_to_string(config, html=False):
    print_ignore = ['weights_dict', 'optimizer_dict']
    args = vars(config)
