import numpy as np
import torch
import torch.nn as nn
from bagging.MergeResults import BaggingResult
from Utils import DataUtils

csv_path = "q1_data/train1.csv"
dataset_path = "q1_data/train.npy"
val_refer_list = "bagging/val.npy"
BATCH_SIZE = 50
CUDA_DEVICE = 2
CLASS_NUM = 20
UP_SIZE = (224,224)

bag_pkl_paths=["./pklmodels/bag0_epoch_24.pkl",
                "./pklmodels/bag1_epoch_24.pkl",
                "./pklmodels/bag2_epoch_24.pkl"]

ValDataset = DataUtils.DatasetLoader(csv_path, dataset_path, refer_list=np.load(val_refer_list),
                                     mode="Valid", up_size=UP_SIZE)
validloader = torch.utils.data.DataLoader(ValDataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
results = BaggingResult(CUDA_DEVICE, bag_pkl_paths=bag_pkl_paths, class_num=CLASS_NUM)


merge_accuracy = []
split_accuracy = [[] for i in range(len(bag_pkl_paths))]
for i, data in enumerate(validloader):
    _, val_x, val_label = data
    merge_res, split_res = results.pred(val_x)
    merge_accuracy.append((val_label==merge_res).numpy().mean())
    print(i*BATCH_SIZE, " - ", (i+1)*BATCH_SIZE)
    
    for j in range(split_res.shape[0]):
        res = split_res[j]
        split_accuracy[j].append((val_label==res).numpy().mean())
    

print("Merge Accuracy: {:.4f}".format(np.array(merge_accuracy).mean()))
for j in range(split_res.shape[0]):
    print("Bag {:d} Accuracy: {:.4f}".format(j, np.array(split_accuracy[j]).mean()))