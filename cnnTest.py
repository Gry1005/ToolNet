#encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


EPOCH = 15   #数据集训练几遍
BATCH_SIZE = 50  #一批数据的个数
LR = 0.001

#加载数据集
path1="../dataSet/train"
#归一化，使颜色值从0-255变为0-1之间
train_data =torchvision.datasets.ImageFolder(path1,transform=transforms.Compose([transforms.Resize((100,100)),transforms.CenterCrop(100),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

#生成训练器
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#加载测试集
path2="../dataSet/test"
test_data = torchvision.datasets.ImageFolder(path2,transform=transforms.Compose([transforms.Resize((100,100)),transforms.CenterCrop(100),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_loader = Data.DataLoader(test_data,batch_size=150,shuffle=True, num_workers=2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #一般来说一个大卷积层包括卷积层，激活函数和池化层
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3, #表示原始图片有多少层，也就是有多少不同种类的特征值，如RGB图片，有红，绿，蓝三个值
                out_channels=16,#表示输出多少个不同种类的特征值；也就是对同一个图片块，有16个过滤器同时工作
                kernel_size=5, #一个过滤器的长和宽都是五个像素点
                stride=1, #相邻两次扫描的图片块之间相隔几个像素点
                padding=2, #在图片周围多出2圈0值，防止过滤器的某一边超过图片边界，如何计算：if stride=1,padding=(kernel_size-1)/2，保证提取出的新图片长宽和原图一样
            ),
            nn.ReLU(),
            #池化层向下筛选需要的部分
            nn.MaxPool2d(
                kernel_size=2, #使用一个长宽为2的池化过滤器
            ),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2), #输入的图片有16层，输出图片有32层
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out=nn.Linear(32*25*25,5) #输入的高度是32，长宽为25，因为经过两次池化；输出为5个不同的值，即0-4

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x) #x中的数据有四个维度:(batch,32,7,7)
        x=x.view(x.size(0),-1) #保留batch,数据变为二维：(batch,32*7*7);因为输出层只接受一维数据作为输入
        output=self.out(x)
        return output

cnn = CNN()

#训练过程
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() #选择误差函数

if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x=Variable(x)
            b_y=Variable(y)

            output=cnn(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%50 == 0:
                for step2,(x2,y2) in enumerate(test_loader):
                    test_x=Variable(x2)
                    test_y=Variable(y2)
                    test_output=cnn(test_x)
                    pred_y=torch.max(test_output,1)[1].data.numpy()
                    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                    print('Epoch: ', epoch, '| train loss: %.6f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

