import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def get_transform(opt):
        transform_list = []
        if opt.resize_or_crop == 'resize_and_crop':
            pass
        elif opt.resize_or_crop == 'crop':
            pass
        elif opt.resize_or_crop == 'scale_width':
            pass
        elif opt.resize_or_crop == 'scale_width_and_crop':
            pass

        if opt.isTrain and not opt.no_flip:
            pass

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

        return transforms.Compose(transform_list)


    def get_posenet_transform(opt, mean_image):
        pass

    def __scale_width(img, target_width):
        pass

    def __subtract_mean(img, mean_image):
        pass

    def __crop_image(img, size, isTrain):
        pass

    def __to_tensor(img):
        pass