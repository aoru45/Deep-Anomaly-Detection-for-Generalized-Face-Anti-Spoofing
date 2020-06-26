import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import random
'''
1. 写一个dataset                √
2. load他并且所有数据提取特征   √
3. 保存特征和标签txt            √
4. t-sne
'''
class MyModel(nn.Module):
    
    def __init__(self):

        super(MyModel,self).__init__()

        feature_extractor = resnet50(pretrained = False)

        self.net = nn.Sequential()

        for name, module in feature_extractor.named_children():

            if not name == "fc":

                self.net.add_module(name, module)
        
        self.net.eval()
        
    def forward(self,x):
        b,c = x.size()[:2]
        x = self.net(x).view(b,-1)

        return x

def glob2(pattern1, pattern2):
    files = glob.glob(pattern1)
    files.extend(glob.glob(pattern2))
    return files
class T_SNEDataset(Dataset):
    def __init__(self, root = "/ssd/xingduan/BCTC_ALL/Data",sub_dirs = ["2D_Plane","2D_Plane_Mask", "3D_Head_Model_Silicone", "3D_Head_Model_Wax", "Half_Mask"]):
        self.root = root
        self.sub_dirs = sub_dirs
        self.pos_filelist = glob2("{}/{}/*_rgb.jpg".format(root, "Live_Person"), "{}/{}/*_ir.jpg".format(root, "Live_Person"))
        self.neg_filelist = []
        for sub_dir in sub_dirs:
            self.neg_filelist.extend(glob2("{}/{}/*_rgb.jpg".format(root, sub_dir), "{}/{}/*_ir.jpg".format(root, sub_dir)))
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    def __getitem__(self,idx):
        p = random.randint(0,1) 
        if p == 0:
            q = random.randint(0, len(self.pos_filelist) - 1)
            l = self.pos_filelist[q].split()
        else:
            q = random.randint(0, len(self.neg_filelist) -1 )
            l = self.neg_filelist[q].split()

        img_path = l[0]

        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        
        img_w, img_h = img.size
    
        ymin,ymax,xmin,xmax = 92, 188, 42, 138 # crop 整张脸

        img = img.crop([xmin,ymin,xmax,ymax])

        img = self.transform(img)

        if "Live_Person" in img_path:
            label = 1 # 正样本
        else:
            label = 0
        return img , torch.tensor(label, dtype = torch.long)
    def __len__(self):
        return len(self.pos_filelist) + len(self.neg_filelist)


if __name__ == "__main__":
    da = T_SNEDataset()
    model = MyModel()
    model.load_state_dict(torch.load("./ckpt/149.pth"))
    model.eval()
    model.cuda(9)
    features, labels = [], []
    
    with torch.no_grad():
        for _ in tqdm(range(1000)):
            idx = random.randint(0, len(da) -1)
            img, label = da[idx]
            img = img.cuda(9)
            label = label.cuda(9)
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)
            feat = model(img)
            features.append(feat.data.cpu())
            labels.append(label.data.cpu())
    features = torch.cat(features, dim = 0).numpy()
    labels = torch.cat(labels, dim = 0).numpy()
    np.savetxt("x.txt",features)
    np.savetxt("labels.txt", labels)
