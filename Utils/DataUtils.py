import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms as tfs
from PIL import Image
import cv2, csv


class DatasetLoader(Dataset):
    
    def __init__(self, csv_path, sets_path, refer_list=[], 
                 img_size=(3,32,32), up_size=(96,96), mode="Train"): # 先从refer_list中查询数据编号，再从sets中读取数据
        self.img_size = img_size
        self.dataset = np.load(sets_path)  # train和val的数据
        self.refer_list = refer_list
        self.mode = mode
        with open(csv_path, 'r') as f:
            csvreader = csv.reader(f)
            originaldata = [i for i in csvreader]
            self.labels = [originaldata[i+1][1] for i in range(len(originaldata)-1)]
            
        self.im_aug = {
            "Train":tfs.Compose([  # train
                        tfs.Resize(up_size),  # 调整大小
                        tfs.RandomHorizontalFlip(),
                        # tfs.RandomVerticalFlip(),
                        # tfs.ColorJitter(brightness=0.5, contrast=0.5),
                        # tfs.RandomRotation(8),  # 随机旋转
                        tfs.ToTensor(),
                        tfs.Normalize([np.mean(self.dataset) / 255], [np.std(self.dataset) / 255])
                        #tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ]),
            "Valid":tfs.Compose([ # valid
                        tfs.Resize(up_size),  # 调整大小
                        tfs.ToTensor(),
                        tfs.Normalize([np.mean(self.dataset) / 255], [np.std(self.dataset) / 255])
                        #tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ]),
            "Test":tfs.Compose([ #test
                        tfs.Resize(up_size),  # 调整大小
                        tfs.ToTensor(),
                        tfs.Normalize([np.mean(self.dataset) / 255], [np.std(self.dataset) / 255])
                        #tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])
        }


    def __getitem__(self, index, img_max=255.0):
        if len(self.refer_list):
            dataset_index = self.refer_list[index]
        else:
            dataset_index = index
        
        img = self.dataset[dataset_index].reshape(3, 32, 32) # 从数据集中取出图像
        label = int(self.labels[dataset_index])
        
        img = img.transpose(1,2,0)
        #img_med = cv2.medianBlur(img,3)#RGBmed(img, 3)
        img_pil = Image.fromarray(img, mode='RGB')
        img_aug = self.im_aug[self.mode](img_pil)

        return (dataset_index, img_aug, label)

    def __len__(self):
        if len(self.refer_list):
            return self.refer_list.shape[0]
        
        return self.dataset.shape[0]


def save_img_3x32x32(img, fp):
    img_t = img.transpose(1,2,0)
    cv2.imwrite(fp, cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR))
    