import numpy as np
import torch
import torch.nn as nn
from bagging.MergeResults import BaggingResult
from Utils import DataUtils

csv_path = "q1_data/train2.csv"
dataset_path = "q1_data/train.npy"
val_refer_list = "bagging/val.npy"
BATCH_SIZE = 200

bag_pkl_paths=["./pklmodels/bag0_epoch_40.pkl",
                "./pklmodels/bag1_epoch_40.pkl",
                "./pklmodels/bag2_epoch_40.pkl"]

ValDataset = DataUtils.DatasetLoader(csv_path, dataset_path, val_refer_list, mode="Valid")
validloader = torch.utils.data.DataLoader(ValDataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
results = BaggingResult(3, bag_pkl_paths=bag_pkl_paths)


merge_accuracy = []
split_accuracy = [[] for i in range(len(bag_pkl_paths))]
for i, data in enumerate(validloader):
    val_x, val_label = data
    merge_res, split_res = results.pred(val_x)
    merge_accuracy.append((val_label==merge_res).numpy().mean())
    
    for j in range(split_res.shape[0]):
        res = split_res[j]
        split_accuracy[j].append((val_label==res).numpy().mean())
    

print("Merge Accuracy: {:.4f}".format(np.array(merge_accuracy).mean()))
for j in range(split_res.shape[0]):
    print("Bag {:d} Accuracy: {:.4f}".format(j, np.array(split_accuracy[j]).mean()))