import numpy as np
import torch
import torch.nn as nn
import cv2
import csv
import ResNet


CUDA_DEVICE_IDX = 2
LR = 0.00001
CLASS_NUM = 100
PKL_MODEL_PATH = './pklmodels/train_epoch_1000.pkl'


torch.cuda.set_device(CUDA_DEVICE_IDX)
resnet = ResNet.resnet34(num_classes=CLASS_NUM).cuda()
# MGANet = torch.nn.DataParallel(MGANet)
resnet.load_state_dict(torch.load(PKL_MODEL_PATH))
resnet.eval()
print("pkl loaded")


test_set = np.load("q1_data/test.npy")
test_y = np.zeros(10000)
for i in range(20):
    test_x = np.array([test_set[index].reshape(3, 32, 32) for index in range(i*500, (i+1)*500)])
    tensor_x = torch.from_numpy(test_x).cuda().float()

    val_y_onehot = resnet(tensor_x).detach().cpu()
    val_y_class = torch.argmax(val_y_onehot, 1).numpy()
    
    test_y[i*500:(i+1)*500] = val_y_class
    print(i*500, "-", (i+1)*500)

print(test_y[9980:])