import torch
import torch.nn as nn

class senet(nn.Module):
    def __init__(self,channels,radio=16):
        super(senet,self).__init__()
        #1.全局平均池化。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #2.两个FC层
        self.fc = nn.Sequential(
            nn.Linear(channels,channels//radio,False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//radio,channels,False),
            nn.Sigmoid()
        )
    def forward(self,x):
        #1.现获取x的shape
        b,c,h,w = x.shape
        #2.自适应平均池化
        y0 = self.avg_pool(x)
        #2.1对自适应平均池化的结果shape做处理
        y0 = y0.view(b,c)
        #3.使用FC
        y1 = self.fc(y0)
        #3.1对结果做处理
        y = y1.view(b,c,1,1)
        return x*y

#测试
model = senet(512)
print(model)
inputs = torch.ones([2,512,32,32])
outputs = model(inputs)
print(outputs)