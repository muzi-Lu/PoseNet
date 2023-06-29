'''
argparse模块是Python内置的用于进行命令行选项与参数解析的模块，argparse模块可以让人轻松编写用户友好的命令行接口，
帮助程序员为模型定义参数。

four steps:
1. 导入argparse包 ——import argparse
2. 创建一个命令行解析器对象 ——创建ArgmentParser()对象
3. 给解析器添加命令行参数 ——调用add.argument() 方法添加参数
4. 解析命令行参数 ——使用parse_args()解析添加的参数
'''

import argparse

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo of argparse')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=4)

    args = parser.parse_args()
    print(args)
    epochs = args.epochs
    batch = args.batchsize
    print('show {}  {}'.format(epochs, batch))
'''

class AbsPoseConfig:
    def __init__(self):
        description = 'Absolute Pose Regression'
        parser = argparse.ArgumentParser(description=description)
        self.parser = parser

        # Add different groups for arguments
        prog_group = parser.add_argument_group('Gerenal Program Config')
        data_group = parser.add_argument_group('Data Loading Config', 'Options regardings image loading and preprocessing')
        model_group = parser.add_argument_group('Model Config', 'Options regarding network model and optimization')
        visdom_group = parser.add_argument_group('Visdom Config', 'Options regarding visdom server for visualization')

        # Program general settings
        prog_group.add_argument('test', action='store_false', dest='testing', help='set program to a testing phase')
        prog_group.add_argument('train', action='store_true', dest='training', help='set program to a training phase')
        prog_group.add_argument('--validate', '-val', metavar='%d[N]', type=int, nargs=1,
                                help='set program to a training phase')
        prog_group.add_argument('--pretrained', metavar='%s', type=str, nargs=1,
                                help='set program to a training phase')
        prog_group.add_argument('--validate', '-val', metavar='%d[N]', type=int, nargs=1,
                                default=None, help='the pretrained weights to initialize the model(default: %s(default)s)')
        prog_group.add_argument('--resume', metavar='%s', type=str, nargs=1, default=None,
                                help='the checkpoint file to reload(default: %s(default)s)')
        prog_group.add_argument('--seed', 's', metavar='%d', type=int, default=1,
                                help='seed for randomization(default)')

