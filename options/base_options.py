"""Object for parsing command line strings into Python objects.

    基础参数的传入与想法
    __init__()
    initialized()
    """

import argparse
import os
import torch
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.isTrain = False

    def initialize(self):
        #self.parser.add_argument('--dataroot', required=True, default='../data', help='path to images')
        self.parser.add_argument('--dataroot', default='../../data', help='path to images')
        self.parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--loadsize', type=int,default=256, help='scale image to this size')
        self.parser.add_argument('--finesize', type=int, default=224, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channel')
        self.parser.add_argument('--output_nc', type=int, default=7, help='of out image channel')
        self.parser.add_argument('--lstm_hidden_size', type=int, default=256, help='hidden size of the LSTM layer in PoseNet+LSTM')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids:e.g. 0 0,1,2 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='test', help='name of experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='posenet', help='chooses how datasets are loaded. [unaligned| aligned | single]')
        self.parser.add_argument('--model', type=str, default='posenet', help='chooses which models to use')
        self.parser.add_argument('--nThreads', type=int, default=8, help='threads to load data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches,otherwise takes them randomsly') # not understand
        self.parser.add_argument('--display_winsize', type=int, default=224, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Max number of samples allowed per dataset') #not understand
        self.parser.add_argument('--resize or crop', type=str, default='scale width or crop', help='scaling and cropping of images at load time [resize and crop |crop| scale width| scale and crop]')
        self.parser.add_argument('--no flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--seed', type=int, default=0, help='initial random seed for deterministic results')
        self.parser.add_argument('--beta', type=float, default=500, help='beta factor used in posenet')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # self.opt.isTrain = self.isTrain # isTrain 怎么传进来的

        # str_ids = self.opt.gpu_ids.spilt(,)  #咋感觉有问题

        args = vars(self.opt) #var变量是啥东西

        print('-------- Options --------')
        for k,v in sorted(args.items()):
            print('%s %s' % (str(k), str(v)))
        print('-------- End --------')

        #save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        #file_name = os.path.join(expr_dir, 'opt_' + self.opt.phase + '.txt') 但是我不知道这个是什么属性诶
        file_name = os.path.join(expr_dir, 'opt_'+self.opt.name+'.txt')

        with open(file_name,'wt') as opt_file:
            opt_file.write('-------- Options --------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s %s\n' % (str(k), str(v)))
            opt_file.write('-------- End --------\n')
        return self.opt



# # # user for test

if __name__ == '__main__':
    train = BaseOptions().parse()