import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
from albumentations import *
import cv2 as cv
import math
import sys
import glob
def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=1, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
        OneOf([
            #CLAHE(clip_limit=2),
            #IAASharpen(),
            #IAAEmboss(),
            RandomBrightnessContrast(0.1,0.1),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)

def glob2(pattern1, pattern2):
    files = glob.glob(pattern1)
    files.extend(glob.glob(pattern2))
    return files
class Traindataset(Dataset):
    def __init__(self,root = "/ssd/xingduan/BCTC_ALL/Data",sub_dirs = ["2D_Plane","2D_Plane_Mask", "3D_Head_Model_Silicone", "3D_Head_Model_Wax", "Half_Mask"]):
        self.root = root
        self.sub_dirs = sub_dirs
        self.pos_filelist = {
            "liveness": glob2("{}/{}/*_rgb.jpg".format(root, "Live_Person"), "{}/{}/*_ir.jpg".format(root, "Live_Person"))
        }
        self.neg_filelist = {
            sub_dir: glob2("{}/{}/*_rgb.jpg".format(root, sub_dir), "{}/{}/*_ir.jpg".format(root, sub_dir)) for sub_dir in sub_dirs
        }
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomGrayscale(0.5),
            transforms.ToTensor()
        ])
        self.aug = strong_aug(0.5)


    def __getitem__(self,idx):
        imgs = []
        labels = None # 规定 0 -> 正正负    1 -> 负负正
        if idx % 2 ==0: # 正正负的情况
            labels = 0
            for k in range(3):
                if k == 0:
                    t = random.randint(0, len(self.pos_filelist["liveness"]) -1)
                    l = self.pos_filelist["liveness"][t].split() # 取一个正样本
                elif k == 1:
                    t = random.randint(0, len(self.pos_filelist["liveness"]) -1)
                    l = self.pos_filelist["liveness"][t].split()
                else:
                    key = random.choice(self.sub_dirs)
                    t = random.randint(0, len(self.neg_filelist[key]) -1)
                    l = self.neg_filelist[key][t].split() # 从所有类型的负样本中随机选取一个
                img_path = l[0]

                img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
                
                img_w, img_h = img.size
            
                ymin,ymax,xmin,xmax = 92, 188, 42, 138 # crop 整张脸

                img = img.crop([xmin,ymin,xmax,ymax])
                
                img = self.aug(image = np.array(img))["image"] #self.transform(img)
    
                img = self.transform(Image.fromarray(img))

                imgs.append(img)
        else: # 负负正的情况

            labels = 1

            for k in range(3):

                if k == 0:
                    key = random.choice(self.sub_dirs)
                    t = random.randint(0, len(self.neg_filelist[key]) -1)
                    l = self.neg_filelist[key][t].split()
                elif k == 1:
                    key = random.choice(self.sub_dirs)
                    t = random.randint(0, len(self.neg_filelist[key]) -1)
                    l = self.neg_filelist[key][t].split()
                    
                else:
                    t = random.randint(0, len(self.pos_filelist["liveness"]) -1)
                    l = self.pos_filelist["liveness"][t].split()
                img_path = l[0]

                img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

                img_w, img_h = img.size
            
                ymin,ymax,xmin,xmax = 92, 188, 42, 138 # crop 整张脸

                img = img.crop([xmin,ymin,xmax,ymax])
                
                img = self.aug(image = np.array(img))["image"] #self.transform(img)
    
                img = self.transform(Image.fromarray(img))

                imgs.append(img)

        return imgs[0], imgs[1], imgs[2], torch.tensor(labels, dtype = torch.long)
    def __len__(self):
        return 20000



if __name__ == "__main__":
    train_dataset = Traindataset()

    for i in range(30):
        anchor, positive, negative, label = train_dataset[0]
        
        img_anchor = (anchor.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        img_positive = (positive.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        img_negative = (negative.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        cv.imshow("img_anchor", cv.cvtColor(img_anchor, cv.COLOR_RGB2BGR))
        cv.imshow("img_positive", cv.cvtColor(img_positive, cv.COLOR_RGB2BGR))
        cv.imshow("img_negative", cv.cvtColor(img_negative, cv.COLOR_RGB2BGR))
        print(label)
        key = cv.waitKey(0)
        if key == ord("q"):
            break
    
    cv.destroyAllWindows()
