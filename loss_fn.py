import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self,margin = 0.2, sigma = 0.3):
        super(TripletLoss,self).__init__()
        self.margin = margin
        self.sigma = sigma
    def forward(self,f_anchor,f_positive, f_negative): # (-1,c)
        d_ap = torch.norm(f_anchor - f_positive, dim = 1) / self.sigma # (-1,1)
        d_an = torch.norm(f_anchor - f_negative, dim = 1) / self.sigma
        return torch.clamp(torch.exp(d_ap) - torch.exp(d_an) + self.margin,0).sum()
        
class MetricSoftmaxLoss(nn.Module):
    def __init__(self):
        super(MetricSoftmaxLoss,self).__init__()

    def forward(self,f_anchor,f_positive, f_negative):
        d_ap = torch.norm(f_anchor - f_positive, dim = 1)
        d_an = torch.norm(f_anchor - f_negative, dim = 1)
        return -torch.log(torch.exp(d_an) / (torch.exp(d_an) + torch.exp(d_ap))).sum()

def hard_samples_mining(f_anchor,f_positive, f_negative, margin):
    d_ap = torch.norm(f_anchor - f_positive, dim = 1)
    d_an = torch.norm(f_anchor - f_negative, dim = 1)
    idx = (d_ap - d_an) < margin
    return idx 
def renorm(x):
    return x.renorm(2,0,1e-5).mul(1e5)
class MetricLoss(nn.Module):
    def __init__(self,margin = 0.2, sigma = 0.3, l = 1.):
        super(MetricLoss, self).__init__()
        self.l = l
        self.margin = margin
        self.trip = TripletLoss(margin, sigma)
        self.soft = MetricSoftmaxLoss()
        
    def forward(self, f_anchor,f_positive, f_negative):
        f_anchor, f_positive, f_negative = renorm(f_anchor), renorm(f_positive), renorm(f_negative)
        with torch.no_grad():
            idx = hard_samples_mining(f_anchor, f_positive, f_negative, self.margin)
            #print(idx)
        loss_trip = self.trip(f_anchor, f_positive, f_negative)
        loss_soft = self.soft(f_anchor, f_positive, f_negative)
        #print(loss_trip.item(), loss_soft.item())
        return loss_trip  + self.l * loss_soft
        #return self.trip(f_anchor[idx], f_positive[idx], f_negative[idx])  + self.l * self.soft(f_anchor[idx], f_positive[idx], f_negative[idx])
        
if __name__ == "__main__":
    x = torch.randn(3,17)
    y = torch.randn(3,17)
    z = torch.randn(3,17)

    loss_fn = MetricLoss()
    res = loss_fn(x,y,z)
