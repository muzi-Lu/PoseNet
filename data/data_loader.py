import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataloader(opt):
    from data.custom_dataset_data_loader import C
