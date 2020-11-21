import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import ResNet
import time, os
import shutil

CUDA_DEVICE_IDX = 2
LR = 0.00001
CLASS_NUM = 100
EPOCH_NUM = 15000
EPOCH_SIZE = 500
VAL_EVERY = 20
SAVE_EVERY = EPOCH_NUM / 5
LOGPATH = "resnet_train.log"

def get_time_stamp():
    return str(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))

def printlog(data, p=1, fp=None):
    if fp==None:
        fp = LOGPATH
    logfile = open(fp, "a")
    seed = np.random.random()
    if seed < p:
        if (logfile != None):
            logfile.write(get_time_stamp() + " " + str(data) + "\n")
        print(get_time_stamp() + " " + str(data))
    logfile.close()

def create_epoch(batchSize):
    dataset_size = train_list.shape[0] # 用于训练的部分大小 50000-2000
    x = np.zeros([batchSize, 3, 32, 32])
    label = np.zeros(batchSize)
    
    for i in range(batchSize):
        train_list_index = int(np.random.random()*dataset_size) # 随机生成一个“训练数据标号列表”的标号
        img_index = train_list[train_list_index] # 得到对应的训练数据标号
        # img_index = i
        x[i] = train_set[img_index].reshape(3, 32, 32) # 从数据集中取出图像
        label[i] = int(labels[img_index])
        
    return x, label

def int2onehot(label):
    batchSize = label.shape[0]
    label_onehot = np.zeros([batchSize, CLASS_NUM])
    for i in range(batchSize):
        label_onehot[i][int(label[i])] = 1
    return label_onehot

def train_step(batchSize):
    resnet.train()
    x, label = create_epoch(batchSize)
    label_onehot = int2onehot(label)
    
    tensor_x = torch.from_numpy(x).cuda().float()
    tensor_label = torch.from_numpy(label).type(torch.LongTensor).cuda()
    
    for j in range(10):
        tensor_y = resnet(tensor_x)
        lossfunc = nn.CrossEntropyLoss()
        loss = lossfunc(tensor_y, tensor_label)
        loss.backward()
        optimizer.step()
        # print(loss)
    
    loss_np = loss.detach().cpu().numpy()
    y = tensor_y.detach().cpu()
    y_class = torch.argmax(y, 1).numpy()
    
    accuracy = float(np.sum([int(label[i]==y_class[i]) for i in range(label.shape[0])])) / batchSize
    return loss_np, accuracy

def val_step(valSize):
    resnet.eval()
    val_label = np.array([int(labels[index]) for index in val_set_list[:valSize]])
    val_x = np.array([train_set[index].reshape(3, 32, 32) for index in val_set_list[:valSize]])
    tensor_x = torch.from_numpy(val_x).cuda().float()

    val_y_onehot = resnet(tensor_x).detach().cpu()
    val_y_class = torch.argmax(val_y_onehot, 1).numpy()
    
    accuracy = float(np.sum([int(val_label[i]==val_y_class[i]) for i in range(val_label.shape[0])])) / val_label.shape[0]
    
    printlog("Val Accuracy: " + str(accuracy))
    

os.remove(LOGPATH)
printlog("Current PID: " + str(os.getpid()))

# 使用split_val_train.py将train.npy数据集拆分成两部分
# 从"train_list.npy"中读取属于train部分的标号
train_list = np.load("train_list.npy")  # 用于训练的数据标号列表
val_set_list = np.load("val_list.npy")  # 用于验证的数据标号列表
train_set = np.load("q1_data/train.npy")  # train和val的数据
with open('q1_data/train2.csv', 'r') as f:
    csvreader = csv.reader(f)
    originaldata = [i for i in csvreader]
    labels = [originaldata[i+1][1] for i in range(len(originaldata)-1)]

torch.cuda.set_device(CUDA_DEVICE_IDX)
resnet = ResNet.resnet34(num_classes=CLASS_NUM).cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=LR, weight_decay=0.005)

for i in range(EPOCH_NUM):
    loss, accuracy = train_step(EPOCH_SIZE)
    printlog("{:d}/{:d} loss: {:.4f}  accuracy: {:.4f}".format(i, EPOCH_NUM, loss, accuracy))
    
    if (i + 1) % VAL_EVERY == 0:
        val_step(EPOCH_SIZE)
    
    if (i + 1) % SAVE_EVERY == 0:
        torch.save(resnet.state_dict(),"./pklmodels/train_epoch_"+str(i+1)+".pkl")
