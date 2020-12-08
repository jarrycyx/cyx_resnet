# 从LOG文件读取数据，并可视化训练进程
import numpy as np
import matplotlib.pyplot as plt
import re

LOG_PATH = "./logs/2020-12-07-204118.log"
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

    val_accus = [float(accu[:6]) for accu in re.findall("Val Accuracy: (.+?) \(" + bag_des, logstr)]
    # print(val_accus, len(val_accus))
    
    return mean_train_accus, mean_train_losses, val_accus
    
train_accuA, train_lossA, val_accuA = get_bag("Class100_A")
train_accuB, train_lossB, val_accuB = get_bag("Class100_B")
train_accuC, train_lossC, val_accuC = get_bag("Class100_C")

epochnum = np.arange(40)
merge_accu = np.ones(40)*0.8822

plt.plot(epochnum, train_accuA, label="Train A")
plt.plot(epochnum, train_accuB, label="Train B")
plt.plot(epochnum, train_accuC, label="Train C")
plt.plot(epochnum, val_accuA, label="Val A")
plt.plot(epochnum, val_accuB, label="Val B")
plt.plot(epochnum, val_accuC, label="Val C")
plt.plot(epochnum, merge_accu,linestyle=":", label="Bagging")
plt.title("Train & Val Accuracy (Classnum=100)")
plt.xlabel("Epoch Num")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(epochnum, train_lossA, label="Train A")
plt.plot(epochnum, train_lossB, label="Train B")
plt.plot(epochnum, train_lossC, label="Train C")
plt.xlabel("Epoch Num")
plt.ylabel("Loss")
plt.title("Train Loss (Classnum=100)")
plt.legend()
plt.show()