import torch
from tqdm import tqdm
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import glob
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet50
import random
import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data.sampler import  WeightedRandomSampler
import numpy as np
import torch.backends.cudnn as cudnn
from loss_fn import MetricLoss
from dataset import Traindataset
cudnn.benchmark = True
device = 3

class MyModel(nn.Module):

    def __init__(self):

        super(MyModel,self).__init__()

        feature_extractor = resnet50(pretrained = True)

        self.net = nn.Sequential()

        for name, module in feature_extractor.named_children():

            if not name == "fc":

                self.net.add_module(name, module)
        
        self.net.train()
        
    def forward(self,x,y,z):
        b,c = x.size()[:2]
        x = self.net(x).view(b,-1)
        y = self.net(y).view(b,-1)
        z = self.net(z).view(b,-1)
        return x,y,z
def train(model,criterion,optimizer,dataloader,num_epoches = 15,scheduler = None):
    model.cuda(device)

    step = 0

    for epoch in range(num_epoches):

        for phase in ["train","save"]:

            if phase == "train":

                running_loss = 0.0

                model.train()
                
                for anchors, positives, negatives, labels in tqdm(dataloader[phase]):
                    
                    anchors = anchors.cuda(device) 

                    positives = positives.cuda(device)

                    negatives = negatives.cuda(device)

                    labels = labels.cuda(device) # 0 for positive positive negative and 1 for negative negative positive
                    
                    feature_a, feature_p, feature_n = model(anchors, positives, negatives)

                    loss = criterion(feature_a, feature_p, feature_n)


                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                    
                    if step % 100 == 0:
                    
                        print("-step: {} -loss: {} ".format(step, loss.item()))
                    
                    step += 1
                print("-epoch:{} -phase:{} -loss:{}".format(epoch,phase,running_loss/len(dataloader[phase])))

                if scheduler is not None:

                    scheduler.step()
            else:
                torch.save(model.state_dict(),"./ckpt/{}.pth".format(epoch))
                pass
                # model.eval()

                # correct = 0

                # with torch.no_grad():

                #     for inputs,targets in tqdm(dataloader[phase]):
                        
                #         inputs = inputs.cuda(device)

                #         targets = targets.cuda(device)

                #         out = model(inputs).data.cpu().numpy() # (n,1)
                        
                #         pred = np.where(out >0.5, 1, 0)

                #         num_correct = (pred == targets.long().data.cpu().numpy()).sum().item()

                #         correct += num_correct

                # acc = correct/len(dataloader[phase].dataset)

                # print("-epoch: {} -phase: {} accuracy: {}".format(epoch,phase,acc))
                
                # if acc >= best_acc:
                    
                #     best_acc = acc

                #     torch.save(model.state_dict(),"./ckpt/{}.pth".format(epoch))

if __name__ == "__main__":
    model = MyModel()#resnet18(pretrained= False)

    criterion = MetricLoss()

    train_dataset = Traindataset()

    dataloader = DataLoader(train_dataset, batch_size = 12, shuffle = True , num_workers=2)

    optimizer = optim.Adam(params = model.parameters(),lr = 0.01)

    train(model, criterion, optimizer, {"train": dataloader}, num_epoches=150)

