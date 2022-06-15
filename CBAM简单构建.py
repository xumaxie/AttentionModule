from msilib.schema import Class
from turtle import forward
from unittest import result
import torch
import torch.nn as nn

#CBAM -->Convolution Block Attention Module
#涉及   通道注意力机制   和   空间注意力机制

#1.通道注意力机制
class channelAtten(nn.Module):
    def __init__(self,channel,radio=16):
        super(channelAtten,self).__init__()
        #1.自适应平均池化和最大池化
        self.maxpool= nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1) #因为目的是得到的w和h为1
        #2.fc层
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//radio,False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//radio,channel,False)
        )
        #3.sigmoid(因为是max和avg的FC相加以后再进行Sigmoid)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        b,c,h,w = x.size()
        #1.进行全局平均池化和全局最大池化
        maxpool = self.maxpool(x).view([b,c])
        avgpool = self.avgpool(x).view([b,c])

        #2.fc
        fcmaxpool = self.fc(maxpool)
        fcavgpool = self.fc(avgpool)
        #3.fc结果相加
        fc = fcavgpool+fcmaxpool
        #4.sigmoid
        result = self.sigmoid(fc)
        #5.修改shape
        result = result.view(b,c,1,1)
        print("通道注意力的结果shape是",result.shape) #通道注意力的结果shape是 torch.Size([2, 512, 1, 1])
        result = result*x
        print("通道注意力*x后为",result.shape)  #通道注意力*x后为 torch.Size([2, 512, 32, 32])
        return result

#2.空间注意力机制
class spatialAtten(nn.Module):
    def __init__(self,kernel=7):
        super(spatialAtten,self).__init__()
        #1.对总的通道上取最大值和平均值
            #区别通道注意力机制：是单个特征层（一个通道）
            #在forward中使用
        #2.卷积层，注意输入通道数为2，输出通道数为1
        padding = kernel//2
        self.conv = nn.Conv2d(2,1,kernel_size=kernel,stride=1,padding=padding)
        #3.sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):  
        #1.1    
        meanpool = torch.mean(x,dim=1,keepdim=True)  
        maxpool,_= torch.max(x,dim=1,keepdim=True)
        print("torch.max后形状是",maxpool.shape)  #torch.max后形状是 torch.Size([2, 1, 32, 32])
        #1.2cat结果
        pool = torch.cat([meanpool,maxpool],dim=1)
        #2.
        conv = self.conv(pool)
        #3.
        result = self.sigmoid(conv)
        print("空间注意力的结果shape是",result.shape)  #空间注意力的结果shape是 torch.Size([2, 1, 32, 32])
        result = result*x
        print("通道注意力*x后为",result.shape)   #通道注意力*x后为 torch.Size([2, 512, 32, 32])
        return result

#3.创建CBAM
class cbam(nn.Module):
    def __init__(self,channel,radio = 16,kernel=7):
        super(cbam,self).__init__()
        self.channelAtten = channelAtten(channel,radio)
        self.spatialAtten = spatialAtten(kernel=kernel)

    def forward(self,x):
        result = self.channelAtten(x)
        result = self.spatialAtten(result)
        return result

model = cbam(512) 
print(model)
inputs = torch.ones([2,512,32,32])
outputs = model(inputs)
# print(outputs)


