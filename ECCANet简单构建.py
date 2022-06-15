import torch
import torch.nn as nn
import math

class ecanet(nn.Module):
    def __init__(self,channel,gamma=2,b=1):
        
        super(ecanet,self).__init__()
        #1.自适应kernel
        #   我们定义kernel的大小，让kernel的大小是与给定的图像动态变化的
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size//2

        #2.设置GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #3.conv层
        self.conv = nn.Conv1d(1,1,kernel_size,padding=padding)
        #4.sigmoid
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b,c,h,w = x.shape
        #1.avg_pool，在view时要注意GAP后应该是序列形式即：一个长条
        avg_pool_out = self.avg_pool(x).view(b,1,c) #c=c*1*1
        #2.conv
        conv_out = self.conv(avg_pool_out)
        #3.sigmoid
        result = self.sigmoid(conv_out)
        #4.view
        result = result.view(b,c,1,1)
        return result*x
model = ecanet(512)
print(model)
inputs = torch.ones([2,512,32,32])
outputs = model(inputs)
print(outputs)
