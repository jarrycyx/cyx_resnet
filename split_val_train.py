import numpy as np
import random

val_list = random.sample(range(0, 50000), 2000)

train_list = []

for i in range(50000):
    if not i in val_list:
        train_list.append(i)
        
train_list = np.array(train_list)

np.save("train_list.npy", train_list)
np.save("val_list", val_list)