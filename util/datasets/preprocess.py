import torch
import random
from torchvision import transforms
import numpy as np
from PIL import Image, ImageChops

'''
ToDO:
1. use those functions to do a lot of experiments to compare what happened and not happened
2. more efficient ways to finished
'''


def get_transform_ops(resize=256, image_mean=None, crop='center', crop_size=224, normalize=False):
    ops = []
    if resize:
        ops.append(transforms.Resize(resize, Image.BICUBIC))
    if image_mean is not None:
        ops.append()
    if crop == 'center':
        ops.append()
    elif crop == 'random':
        ops.append()
    if normalize:
        ops.append()
        ops.append()
    else:
        ops.append()
    return transforms.Compose(ops)


class ToTensorUnscaled(object):
    '''
    convert a RGB Image to a CHW ordered Tensor
    '''

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1))  # HWC --> CHW

    def __repr__(self):
        return 'ToTensorUnscaled'


class ToTensorScaled(object):
    '''
    convert a RGB Image to a CHW ordered Tensor
    '''

    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        im /= 255.0
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1))  # HWC --> CHW

    def __repr__(self):
        return 'ToTensorScaled'


class MeanSubtractPIL(object):
    '''
    Mean subtract operates on PIL Images
    '''

    def __init__(self, im_mean):
        self.im_mean = im_mean

    def __call__(self, im):
        if self.im_mean is None:
            return im
        return ImageChops.subtract(im, self.im_mean)


class MeanSubtractNumpy(object):
    '''
    Mean subtract operates on numpy ndarrays
    '''

    def __init__(self, im_mean):
        self.im_mean = im_mean

    def __call__(self, im):
        if self.im_mean is None:
            return im
        return np.array(im).astype('float') - self.im_mean.astype('float')

    def __repr__(self):
        if self.im_mean is None:
            return 'MeanSubtractNumpy(None)'
        return 'MeanSubtractNumpy(im_mean={})'.format(self.in_mean.shape)


class CenterCropNumpy(object):
    def __init__(self, size):
        pass

    def __call__(self, im):
        pass

    def __repr__(self):
        pass


class RandomCropNumpy(object):
    def __init__(self):
        pass

    def __call__(self, im):
        pass

    def __repr__(self):
        pass
