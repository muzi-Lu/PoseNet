import torch
import random
import numpy as np
from collections import OrderedDict

def load_weights_to_gpu(weights_dir=None, gpu=None):
    pass

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # 没有
    torch.backends.cudnn.benchmark = False # 没有阿