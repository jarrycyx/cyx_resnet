from TrainBagging import TrainBag
import threading
from Utils.LogUtils import Log
import numpy as np
import torch

class TrainThread (threading.Thread):
    def __init__(self, name, trainbag):
        threading.Thread.__init__(self)
        self.name = name
        self.trainbag = trainbag
        
    def run(self):
        log.printlog("Start Thread: " + self.name)
        init_train(self.trainbag, self.name)
        log.printlog ("Exit Thread: " + self.name)

def init_train(trainbag, name):
    accu = []
    for i in range(EPOCH_NUM):
        log.printlog("Epoch: {:d}/{:d} ({:s})".format(i, EPOCH_NUM, name))
        trainbag.train_step(show_every=200)
        accu.append(trainbag.val_step())
        if (i+1) % int(EPOCH_NUM/4) == 0:
            torch.save(trainbag.resnet.state_dict(),"./pklmodels/"+name+"_epoch_"+str(i+1)+".pkl")
            log.printlog("Saving state pkls:" + name)
            
    np.save("logs/"+name+"_accu.npy", np.array(accu))
        

CUDA_DEVICE = [2,3,3]
DESCRIPTIONS = ["Class100_A", "Class100_B", "Class100_C"] # different descriptions
BAGS_NPY = ["bagging/bag1.npy", "bagging/bag2.npy", "bagging/bag3.npy"]
EPOCH_NUM = 40
log = Log(clear=True)

trainbags = []
threads = []

for i in range(len(CUDA_DEVICE)):
    trainbags.append(TrainBag("q1_data/train2.csv", "q1_data/train.npy", BAGS_NPY[i], "bagging/val.npy", logUtil=log, cuda_device=CUDA_DEVICE[i], description=DESCRIPTIONS[i]))
    trainbags[i].load_net(epoch_num=EPOCH_NUM)
    threads.append(TrainThread(DESCRIPTIONS[i], trainbags[i]))
    threads[i].start()

for thread in threads:
    thread.join()
    
log.printlog("Threads Ended")