import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import time, os
import shutil
from torchvision import transforms as tfs
from PIL import Image
from torchvision import models

from Utils.LogUtils import Log
from Utils.ImgProcessing import ImgAugment
from Utils import DataUtils
from loss.MultiLoss import MultiLoss



class TrainBag(object):

    CUDA_DEVICE_IDX = 2
    LR = 0.00002
    CLASS_NUM = 100
    BATCH_SIZE = 30
    LOGPATH = "resnet_train.log"
    WEIGHT_DECAY = 0

    UP_SIZE = (224,224)
    SAVE_IMG = False
    
    def printlog(self, str):
        if not self.log == None:
            self.log.printlog(str)
    
    def __init__(self, csv_path, dataset_path, bag_refer_list, val_refer_list, logUtil=None, cuda_device=2, description=""):
        self.log = logUtil
        self.printlog("Current PID: " + str(os.getpid()))
        self.device = cuda_device
        self.description = description
        
        TrainDataset = DataUtils.DatasetLoader(csv_path, dataset_path, refer_list=np.load(bag_refer_list),
                                               mode="Train", up_size=self.UP_SIZE)
        ValDataset = DataUtils.DatasetLoader(csv_path, dataset_path, refer_list=np.load(val_refer_list), 
                                             mode="Valid", up_size=self.UP_SIZE)
        
        self.trainloader = torch.utils.data.DataLoader(TrainDataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(ValDataset, batch_size=self.BATCH_SIZE, num_workers=2, shuffle=True)
    
    def load_net(self, epoch_num=40):
        # torch.cuda.set_device(self.device)
        # resnet = classnet.ClassNet(num_classes=CLASS_NUM).cuda()
        self.resnet = models.resnet152(pretrained=True)
        fc_in = self.resnet.fc.in_features  # 获取全连接层的输入特征维度
        self.resnet.fc = nn.Linear(fc_in, self.CLASS_NUM)
        self.resnet.to(self.device)
        
        # self.optimizer = torch.optim.SGD(self.resnet.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.optimizer = torch.optim.Adam(self.resnet.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.variableLR = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                   milestones=[int(epoch_num*2/4), int(epoch_num*3/4)], gamma=0.1)
    
    def train_step(self, show_every=30):
        self.resnet.train()
        # lossfunc = nn.CrossEntropyLoss()
        lossfunc = MultiLoss(focal_gamma=2)
        this_LR = self.optimizer.param_groups[0]['lr']
        for i, data in enumerate(self.trainloader):
            _, x, label = data
            tensor_x = x.to(self.device).float()
            tensor_label = label.type(torch.LongTensor).to(self.device)
            
            self.optimizer.zero_grad()
            tensor_y = self.resnet(tensor_x)
            loss = lossfunc(tensor_y, tensor_label)
            loss.backward()
            self.optimizer.step()
            # print(loss)
            
            loss_np = loss.detach().cpu().numpy()
            y = tensor_y.detach().cpu()
            y_class = torch.argmax(y, 1).numpy()
        
            accuracy = np.mean(label.numpy()==y_class)
            
            if i % show_every == 0:
                self.printlog("Batch: {:d}/{:d} loss: {:.4f} accuracy: {:.4f} lr: {:.7f} ({:s})" 
                     .format(i, len(self.trainloader), loss, accuracy, this_LR, self.description))
        
        
        self.variableLR.step()
            
            

    def val_step(self):
        self.resnet.eval()
        accuracy = []
        for i, data in enumerate(self.validloader):
            _, val_x, val_label = data
            
            tensor_x = val_x.to(self.device).float()
            val_y_onehot = self.resnet(tensor_x).detach().cpu()
            val_y_class = torch.argmax(val_y_onehot, 1).numpy()
            accuracy.append(np.mean(val_label.numpy()==val_y_class))
            
            # if 0:
            #     cv2.imwrite("imgs/test.jpg", cv2.cvtColor(val_x[0].transpose(1,2,0), cv2.COLOR_RGB2BGR))    
            
        self.printlog("Val Accuracy: {:.4f} ({:s})".format(np.array(accuracy).mean(), self.description))
    

if __name__ == "__main__":

    CUDA_DEVICE = [2,2,2]
    DESCRIPTIONS = ["Class100_A", "Class100_B", "Class100_C"] # different descriptions
    EPOCH_NUM = 40
    log = Log(clear=True)

    trainbags = []

    trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag1.npy", "bagging/val.npy", logUtil=log, cuda_device=CUDA_DEVICE[0], description=DESCRIPTIONS[0]))
    trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag2.npy", "bagging/val.npy", logUtil=log, cuda_device=CUDA_DEVICE[1], description=DESCRIPTIONS[1]))
    trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", "bagging/bag3.npy", "bagging/val.npy", logUtil=log, cuda_device=CUDA_DEVICE[2], description=DESCRIPTIONS[2]))

    # trainbags.append(TrainBag("q1_data/train1.csv", "q1_data/train.npy", "bagging/train_list.npy", "bagging/val_list.npy", log))
    for j in range(len(trainbags)):
        trainbags[j].load_net(epoch_num=EPOCH_NUM)
        
    for i in range(EPOCH_NUM):
        for j in range(len(trainbags)):
            log.printlog("Bag: {:d} Epoch: {:d}/{:d}".format(j, i, EPOCH_NUM))
            trainbags[j].train_step(show_every=100)
            trainbags[j].val_step()
            if (i+1) % int(EPOCH_NUM/4) == 0:
                torch.save(trainbags[j].resnet.state_dict(),"./pklmodels/"+DESCRIPTIONS[j]+"_epoch_"+str(i+1)+".pkl")
                log.printlog("Saving state pkls")