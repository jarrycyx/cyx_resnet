import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import ResNet


CUDA_DEVICE_IDX = 2
LR = 0.00001
CLASS_NUM = 100
PKL_MODEL_PATH = './pklmodels/train_epoch_5000.pkl'


torch.cuda.set_device(CUDA_DEVICE_IDX)
resnet = ResNet.resnet34(num_classes=CLASS_NUM).cuda()
# MGANet = torch.nn.DataParallel(MGANet)
resnet.load_state_dict(torch.load(PKL_MODEL_PATH))
resnet.eval()
print("pkl loaded")


train_set = np.load("q1_data/train.npy")
val_set_list = np.load("val_list.npy")
with open('q1_data/train2.csv', 'r') as f:
    csvreader = csv.reader(f)
    originaldata = [i for i in csvreader]
    labels = [originaldata[i+1][1] for i in range(len(originaldata)-1)]


val_label = np.array([int(labels[index]) for index in val_set_list])
val_y = np.zeros(2000)
for i in range(4):
    test_x = np.array([train_set[index].reshape(3, 32, 32) for index in val_set_list[i*500:(i+1)*500]])
    tensor_x = torch.from_numpy(test_x).cuda().float()

    val_y_onehot = resnet(tensor_x).detach().cpu()
    val_y_class = torch.argmax(val_y_onehot, 1).numpy()
    
    val_y[i*500:(i+1)*500] = val_y_class
    print(i*500, "-", (i+1)*500)


accuracy = float(np.sum([int(val_label[i]==val_y[i]) for i in range(val_label.shape[0])])) / val_label.shape[0]

print("Val Accuracy: " + str(accuracy))