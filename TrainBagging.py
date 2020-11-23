import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import ResNet
import classnet
import time, os
import shutil
from torchvision import transforms as tfs
from PIL import Image
from torchvision import models

from Utils.LogUtils import Log
from Utils.ImgProcessing import ImgAugment
from Utils import DataUtils



class TrainBag(object):

    CUDA_DEVICE_IDX = 2
    LR = 0.00001
    CLASS_NUM = 100
    BATCH_SIZE = 200
    LOGPATH = "resnet_train.log"
    WEIGHT_DECAY = 0

    UP_SIZE = (96,96)
    SAVE_IMG = False
    
    def __init__(self, csv_path, dataset_path, bag_refer_list, val_refer_list, logUtil):
        self.log = logUtil
        self.log.printlog("Current PID: " + str(os.getpid()))
        
        TrainDataset = DataUtils.DatasetLoader(csv_path, dataset_path, bag_refer_list, mode="Train")
        ValDataset = DataUtils.DatasetLoader(csv_path, dataset_path, val_refer_list, mode="Valid")
        
        self.trainloader = torch.utils.data.DataLoader(TrainDataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(ValDataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True)
    
    def load_net(self, cuda_device_index, epoch_num):
        torch.cuda.set_device(cuda_device_index)
        # resnet = classnet.ClassNet(num_classes=CLASS_NUM).cuda()
        self.resnet = models.resnet50(pretrained=True)
        fc_in = self.resnet.fc.in_features  # 获取全连接层的输入特征维度
        self.resnet.fc = nn.Linear(fc_in, self.CLASS_NUM)
        self.resnet.cuda()
        
        
        self.optimizer = torch.optim.Adam(self.resnet.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.variableLR = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                    milestones=[int(epoch_num*1/3), int(epoch_num*2/3)], gamma=0.1)
    
    def train_step(self, show_every=10):
        self.resnet.train()
        lossfunc = nn.CrossEntropyLoss()
        this_LR = self.optimizer.param_groups[0]['lr']
        for i, data in enumerate(self.trainloader):
            x, label = data
            tensor_x = x.cuda().float()
            tensor_label = label.type(torch.LongTensor).cuda()
            
            self.optimizer.zero_grad()
            tensor_y = self.resnet(tensor_x)
            loss = lossfunc(tensor_y, tensor_label)
            loss.backward()
            self.optimizer.step()
            # print(loss)
            
            loss_np = loss.detach().cpu().numpy()
            y = tensor_y.detach().cpu()
            y_class = torch.argmax(y, 1).numpy()
        
            accuracy = float(np.sum([int(label[i]==y_class[i]) for i in range(label.shape[0])])) / label.shape[0]
            
            if i % show_every == 0:
                self.log.printlog("Batch: {:d}/{:d} loss: {:.4f} accuracy: {:.4f} lr: {:.7f}" 
                     .format(i, len(self.trainloader), loss, accuracy, this_LR))
        
        
        self.variableLR.step()
            
            

    def val_step(self):
        self.resnet.eval()
        accuracy = []
        for i, data in enumerate(self.validloader):
            val_x, val_label = data
            
            tensor_x = val_x.cuda().float()

            val_y_onehot = self.resnet(tensor_x).detach().cpu()
            val_y_class = torch.argmax(val_y_onehot, 1).numpy()
            
            accuracy.append(float(np.sum([int(val_label[i]==val_y_class[i]) for i in range(val_label.shape[0])])) / val_label.shape[0])
            
            # if 0:
            #     cv2.imwrite("imgs/test.jpg", cv2.cvtColor(val_x[0].transpose(1,2,0), cv2.COLOR_RGB2BGR))    
            
        self.log.printlog("Val Accuracy: {:.4f}".format(np.array(accuracy).mean()))
    


EPOCH_NUM = 60
log = Log(clear=True)

trainbags = []

trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag1.npy", "bagging/val.npy", log))
trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag2.npy", "bagging/val.npy", log))
trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag3.npy", "bagging/val.npy", log))
trainbags[0].load_net(2, EPOCH_NUM)
trainbags[1].load_net(2, EPOCH_NUM)
trainbags[2].load_net(2, EPOCH_NUM)

for i in range(EPOCH_NUM):
    for j in range(len(trainbags)):
        log.printlog("Bag: {:d} Epoch: {:d}/{:d}".format(j, i, EPOCH_NUM))
        trainbags[j].train_step()
        trainbags[j].val_step()
        if i % int(EPOCH_NUM/3) == 0:
            torch.save(trainbags[j].resnet.state_dict(),"./pklmodels/bag1_epoch_"+str(i+1)+".pkl")
        
