import torch.nn as nn
from base.base_net import BaseNet

class CustomNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = 32
        self.L =  nn.Sequential(
            nn.Linear(32 * 32 * 3,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.rep_dim)
        )
    def forward(self,x,ys = None):
        x = x.view(x.size()[0],-1)
        return self.L(x)
