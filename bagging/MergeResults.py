import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class BaggingResult(object):
    def __init__(self, device, bag_pkl_paths=None, class_num=100):
        if bag_pkl_paths == None:
            self.bag_pkl_paths = ["../pklmodels/bag0_epoch_80.pkl",
                                    "../pklmodels/bag1_epoch_80.pkl",
                                    "../pklmodels/bag2_epoch_80.pkl"]
        else:
            self.bag_pkl_paths = bag_pkl_paths
        self.CLASS_NUM = class_num
        torch.cuda.set_device(device)
        self.nets = []
        for path in bag_pkl_paths:
            resnet = models.resnet152(pretrained=True)
            fc_in = resnet.fc.in_features  # 获取全连接层的输入特征维度
            resnet.fc = nn.Linear(fc_in, self.CLASS_NUM)
            resnet.load_state_dict(torch.load(path))
            resnet.cuda()
            self.nets.append(resnet)
            print("PKL Loaded")

    def pred(self, batch):
        ys = torch.zeros([len(self.nets), batch.shape[0], self.CLASS_NUM])
        y_sum = torch.zeros([batch.shape[0], self.CLASS_NUM])
        for i in range(len(self.nets)):
            self.nets[i].eval()
            tensor_y = self.nets[i](batch.cuda().float())
            ys[i] = tensor_y.detach().cpu()
            y_sum += ys[i]
        
        _, y_class = torch.max(y_sum, 1)
        _, ys_class = torch.max(ys, 2)
        return y_class, ys_class
        