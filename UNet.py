import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torchvision.transforms as transforms


class Unet(nn.Module):
    def __init__(self,output):
        super(Unet, self).__init__()
        self.cbr_enc = []
        self.cbr_dec = []
        self.up_sample = []
        for i in range(5):
            if i==0:
                self.cbr_enc.append(CBR(1,output*(2**i)))
            else:
                self.cbr_enc.append(CBR(output*(2**(i-1)),output*(2**i)))
        for i in range(4):
            self.up_sample.append(nn.ConvTranspose2d(output*(2**(4-i)),output*(2**(4-i-1)),2,2,0))
        for i in range(4):
            self.cbr_dec.append(CBR(output*(2**(4-i)),output*(2**(4-i-1))))
        self.down_sample = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(output,2,1,1,0)
    def forward(self,x):
        x1 = self.cbr_enc[0](x)
        x2 = self.cbr_enc[1](self.down_sample(x1))
        x3 = self.cbr_enc[2](self.down_sample(x2))
        x4 = self.cbr_enc[3](self.down_sample(x3))
        x5 = self.cbr_enc[4](self.down_sample(x4))
        x6 = torch.cat([x4,self.up_sample[0](x5)],dim=1)
        x7 = torch.cat([x3,self.up_sample[1](self.cbr_dec[0](x6))],dim=1)
        x8 = torch.cat([x2,self.up_sample[2](self.cbr_dec[1](x7))],dim=1)
        x9 = torch.cat([x1,self.up_sample[3](self.cbr_dec[2](x8))],dim=1)
        output = self.conv1_1(self.cbr_dec[3](x9))
        return output

class CBR(nn.Module):
    def __init__(self,input,output):
        super(CBR, self).__init__()
        self.l = nn.Sequential(nn.Conv2d(input,output,1,padding=0),
                               nn.BatchNorm2d(num_features=output,eps=1e-05),
                               nn.ReLU(),
                               nn.Conv2d(output,output,1,padding=0),
                               nn.BatchNorm2d(num_features=output, eps=1e-05),
                               nn.ReLU(),
                               )
    def forward(self,x):
        return self.l(x)

unet=Unet(64)
x=torch.rand(1,1,512,512)
# y=unet(x)
with torch.no_grad():
    torch.onnx.export(unet,x,"test1.onnx",verbose=True,opset_version=11,input_names=['input'],output_names=['output'])