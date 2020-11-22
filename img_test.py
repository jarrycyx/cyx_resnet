import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import ResNet
import classnet
import time, os
import shutil
import scipy.signal as signal

shutil.rmtree("imgs/")
os.mkdir("imgs/")
train_set = np.load("q1_data/train.npy")  # train和val的数据

def save_img(idx):
    print(idx)
    x = train_set[idx].reshape(3, 32, 32) # 从数据集中取出图像
    img = x.transpose(1,2,0)
    img_med = cv2.medianBlur(img,3)#RGBmed(img, 3)
    cv2.imwrite("imgs/"+str(idx)+".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("imgs/"+str(idx)+"_med.jpg", cv2.cvtColor(img_med, cv2.COLOR_RGB2BGR))

def save_imgs(num):
    dataset_size = train_set.shape[0] # 用于训练的部分大小 50000-2000
    
    for i in range(num):
        idx = int(np.random.random()*dataset_size)
        save_img(idx)
        
save_img(10006)
save_img(10009)
save_img(10013)