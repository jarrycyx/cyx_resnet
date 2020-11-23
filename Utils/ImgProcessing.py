import numpy as np
import torch
from torchvision import transforms as tfs
from PIL import Image
import cv2


class ImgAugment(object):
    
    def __init__(self, up_size=(100,100)):
        self.im_aug = [tfs.Compose([  # train
                    tfs.RandomHorizontalFlip(),
                    tfs.RandomVerticalFlip(),
                    tfs.ColorJitter(brightness=0.5, contrast=0.5),
                    tfs.RandomRotation(45),  # 随机旋转
                    tfs.Resize(up_size),  # 调整大小
                    tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]),
                tfs.Compose([ # valid
                    tfs.Resize(up_size),  # 调整大小
                    tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]),
                tfs.Compose([ #test
                    tfs.Resize(up_size),  # 调整大小
                    tfs.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ])]
            

    def imgPreprcs(self, img, mode=0, max=255.0): # 3x32x32, 0:train, 1:valid, 2:test
        img = img.transpose(1,2,0).astype(np.float)/max
        img_med = cv2.medianBlur(img,3)#RGBmed(img, 3)
        img_pil = Image.fromarray(img_med, mode='RGB')
        img_aug = self.im_aug[mode](img_pil)
        
        return np.asarray(img_aug).transpose(2,0,1)