#encoding:utf-8
import torch
import torch.nn as nn
import torch.hub
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt

#多加入三个全连接层

EPOCH = 20   #数据集训练几遍
BATCH_SIZE = 48  #一批数据的个数
LR = 0.001



#显示图片
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#加载数据集
path1="../dataSet5/train"
#归一化，使颜色值从0-255变为0-1之间
train_data =torchvision.datasets.ImageFolder(path1,transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
#train_data =torchvision.datasets.ImageFolder(path1,transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]))

#生成训练器
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
#train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#加载测试集
path2="../dataSet5/test"
test_data = torchvision.datasets.ImageFolder(path2,transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=False, num_workers=2)
#test_loader = Data.DataLoader(test_data,batch_size=30,shuffle=True)

#模型结构
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 不同的vgg结构，这样写可以有效节约代码空间。

class VGG(nn.Module):
    # nn.Module是一个特殊的nn模块，加载nn.Module，这是为了继承父类
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # super 加载父类中的__init__()函数
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512*7*7, 6)
        # 该网络输入为Cifar10数据集，因此输出为（512，1，1）

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # 这一步将out拉成out.size(0)的一维向量
        out = self.classifier(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,
                                     padding=1, bias=False),
                           #nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

#加载模型
cnn = VGG('VGG11')
cnn=cnn.cuda()

#训练过程
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() #选择误差函数

#保存网络
def saveNet():
    torch.save(cnn, 'VGG11-3.pkl')

if __name__ == '__main__':
    accuracyList = []
    epochlist=[]
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x=Variable(x).cuda()
            b_y=Variable(y).cuda()

            #imshow(b_x[0,2],"tensorPic")

            output=cnn(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%300 == 0:

                saveNet()
                sum_accuracy = 0.0
                num = 0


                for step2,(x2,y2) in enumerate(test_loader):
                    test_x=Variable(x2).cuda()
                    test_y=Variable(y2).cuda()
                    test_output=cnn(test_x)
                    pred_y=torch.max(test_output,1)[1].cuda().data
                    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                    sum_accuracy = sum_accuracy + accuracy
                    num = num + 1

                    if num%200==0:
                        print('Epoch: ', epoch, '| train loss: %.20f' % loss.item(), '| test accuracy: %.4f' % accuracy)

                print('Epoch: ', epoch, "| total accuracy:%.4f" % (sum_accuracy / num))
                epochlist.append(epoch)
                accuracyList.append([sum_accuracy / num])

    plt.plot(epochlist,accuracyList)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
