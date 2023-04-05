from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os
import collections

# # Converts a Tensor into a Numpy array
# # |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1 : #行向量等于1 # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1,2,0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


'''
def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #dataLoader中设置的mean参数
    std = [0.229,0.224,0.225]  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)
'''


def diagnose_network(net, name='network'):
    mean = 0.0
    count =0
    for param in net.parameters():
        if param is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count = count + 1
    if count > 1:
        mean = mean + count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)  #将image_numpy transform to image
    image_pil.save(image_path)
'''
def info(object, spacing=10, collapse=1):

'''

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)