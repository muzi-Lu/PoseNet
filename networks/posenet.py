import torch
import torch.nn as nn
from collections import OrderedDict

class BaseNet(nn.Module):
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = config
        self.device = config.device

    @property
    def num_params(self):
        return sum([p.numel for p in self.parameters()])

    def forward(self, *inputs):
        """
        Defines the computation performed at every call
        Inherited from superclass torch.nn.Module
        Should be overridden by all subclases
        :param inputs:
        :return:
        """
        raise NotImplementedError

    def get_inputs(self, batch, **kwargs):
        """
        Define how to initialize the weights of the networks.
        Should be overridden by all subclasses, since it normally
        differs according to te network models
        :param batch:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def loss_(self, batch):
        """
        Define how to calculate the loss for the network.
        Should be overridden by all subclasses, since different
        applications or network models may have different types
        of targets and the corresponding criterions to evaluate
        the predictions
        :param batch:
        :return:
        """
        raise NotImplementedError

    def set