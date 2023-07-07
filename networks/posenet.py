import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.basenet import BaseNet
from networks.base.googlenet import GoogleNet

class Regression(nn.Module):
    """
    Pose regression module
    Args:
        regid: id to map the length of the last dimension of the input
        feature maps
        with embedding: if set True, output activations before pose regression
        together with regressed poses, otherwise only poses
    Return:
        xyz: global camera position
        wpqr: global camera orientation in quaternion
    """

    def __init__(self, regid, with_embedding=False):
        super(Regression, self).__init__()
        conv_in = {'regress1': 512, 'regress2':528}
        self.with_embedding = with_embedding
        if regid != 'regress3':
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.7))
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(2048, 3)
            self.regress_fc_wpqr = nn.Linear(2048, 4)
