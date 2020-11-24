import numpy as np
import torch
import torch.nn as nn
from bagging.MergeResults import BaggingResult
from Utils import DataUtils
import math, cv2, csv

from Utils import DataUtils

save_csv_path = "q1_data/samplesummision_class20.csv"
testset_path = "q1_data/test.npy"
BATCH_SIZE = 200
CUDA_DEVICE = 2
CLASS_NUM = 20
UP_SIZE = (224,224)

csvheader = ["image_id", "coarse_label"]

bag_pkl_paths=["./pklmodels/bag0_epoch_40.pkl",
                "./pklmodels/bag1_epoch_40.pkl",
                "./pklmodels/bag2_epoch_40.pkl"]

testDataset = DataUtils.DatasetLoader(save_csv_path, testset_path, 
                                      mode="Test", up_size=UP_SIZE)

setsize = len(testDataset)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
results = BaggingResult(CUDA_DEVICE, bag_pkl_paths=bag_pkl_paths, class_num=CLASS_NUM)

resdata = -np.ones(setsize) # Default label -1
for i, data in enumerate(testloader):
    index, val_x, val_label = data
    print(np.min(index.numpy()), " - ", np.max(index.numpy()))
    
    merge_res, split_res = results.pred(val_x)
    resdata[index] = merge_res.numpy()

csvdata = [(i, int(resdata[i])) for i in range(resdata.shape[0])]

with open(save_csv_path,'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(csvheader)
    f_csv.writerows(csvdata)