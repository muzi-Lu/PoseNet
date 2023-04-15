import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned_posenet':
        pass
    else:
        raise ValueError("Dataset [%s] not recognized" % opt.dataset_mode)