import os
import torch

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self,input):
        pass

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimizer_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        pass

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        pass

    def save_network(self, network, network_label, epoch_label):
        pass

    def update_learning_rate(self):
        pass