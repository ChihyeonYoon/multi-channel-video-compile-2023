import torch
from torchvision.models import swin_b, Swin_B_Weights
import torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class swin_binary_module(nn.Module):
    def __init__(self):
        super(swin_binary_module, self).__init__()
        self.Lin_1 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.Lin_1(x)
        x = self.softmax(x)

        return x
    
class swin_binary(nn.Module):
    def __init__(self, pretrained=False):
        super(swin_binary, self).__init__()
        # self.model = swin_b()
        if pretrained:
            self.model = swin_b(weights = Swin_B_Weights.IMAGENET1K_V1)
            self.model.head = swin_binary_module()
            
        if not pretrained:
            self.model = swin_b()
            self.model.head = swin_binary_module()
        self.loss = CrossEntropyLoss().cuda()

    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':
    model = swin_binary(pretrained=True)
    # print(model)