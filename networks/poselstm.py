import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.basenet import BaseNet
from networks.base.googlenet import GoogleNet

class FourDirectionalLSTM(nn.Module):
    '''
    四方向LSTM是LSTM架构的一种变体，它包含四个独立的LSTM层，每个层以不同的方向处理输入。
    在常规的LSTM中，输入序列按照顺序从左或者从右往左进行处理，然而在四方向LSTM中，输入序列
    被分为四个部分，每个部分由独立的LSTM进行处理
    '''
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()
        self.feat_size = origin_feat_size // seq_size
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.lstm_rightleft = nn.LSTM(self.feat_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_updown = nn.LSTM(self.seq_size, self.hidden_size, batch_first=True, bidirectional=True)


    def init_hidden_(self, batch_size, device):
        '''
        Return initialized hidden states and cell states for each biodirectional lstm cell
        :param batch_szie:
        :param device:
        :return:
        '''
        return (torch.randn(2, batch_size, self.hidden_size).to(device),
                torch.randn(2, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        batch_size = x.size()[0]

class Regression(nn.Module):
    def __init__(self, regid):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
            self.lstm4dir = FourDirectionalLSTM(seq_size=32, origin_feat_size=1024, hidden_size=256)
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            pass
