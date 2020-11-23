import time, os, shutil
import numpy as np


class Log(object):
    def __init__(self, path=None, clear=False):
        if clear:
            if  os.path.exists("./logs/"):
                shutil.rmtree("./logs")
                
            os.mkdir("./logs")
            
        if path:
            self.LOG_PATH = path
        else:
            self.LOGPATH = "./logs/"+self.get_time_stamp()+".log"
        
    def get_time_stamp(self):
        return str(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))

    def printlog(self, data, p=1, fp=None): # p: possibility for this log to be printed
        if fp==None:
            fp = self.LOGPATH
        logfile = open(fp, "a")
        seed = np.random.random()
        if seed < p:
            if (logfile != None):
                logfile.write(self.get_time_stamp() + " " + str(data) + "\n")
            print(self.get_time_stamp() + " " + str(data))
        logfile.close()