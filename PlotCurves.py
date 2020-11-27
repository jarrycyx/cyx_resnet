# 从LOG文件读取数据，并可视化训练进程
import numpy as np
import matplotlib.pyplot as plt
import re

LOG_PATH = "./logs/2020-11-26-200401.log"
VALID_EVERY = 6

with open(LOG_PATH, "r") as f:
    logstr = f.read()

def get_bag(bag_des):
    trainlogs = re.findall("Batch(.+?)" +  bag_des, logstr)
    train_accuracies = [float(re.findall("accuracy: (.+?) ", log)[0]) for log in trainlogs]
    train_losses = [float(re.findall("loss: (.+?) ", log)[0]) for log in trainlogs]

    mean_train_accus = [np.mean(train_accuracies[i*6:(i+1)*6]) for i in range(int(len(train_accuracies)/6))]
    mean_train_losses = [np.mean(train_losses[i*6:(i+1)*6]) for i in range(int(len(train_losses)/6))]

    # print(mean_train_accus, len(mean_train_accus))
    # print(mean_train_losses, len(mean_train_losses))

    val_accus = [float(accu) for accu in re.findall("Val Accuracy: (.+?) \(" + bag_des, logstr)]
    # print(val_accus, len(val_accus))
    
    return mean_train_accus, mean_train_losses, val_accus
    
train_accuA, train_lossA, val_accuA = trget_bag("Class20_A")
