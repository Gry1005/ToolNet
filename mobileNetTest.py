import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 20   #数据集训练几遍
BATCH_SIZE = 64  #一批数据的个数
LR = 0.001

#加载数据集
path1="../dataSet/train"
#归一化，使颜色值从0-255变为0-1之间
train_data =torchvision.datasets.ImageFolder(path1,transform=transforms.Compose([transforms.Resize((32,32)),transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

#生成训练器
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#加载测试集
path2="../dataSet/test"
test_data = torchvision.datasets.ImageFolder(path2,transform=transforms.Compose([transforms.Resize((32,32)),transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_loader = Data.DataLoader(test_data,batch_size=32,shuffle=False, num_workers=2)


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d\
            (in_planes, in_planes, kernel_size=3, stride=stride,
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d\
            (in_planes, out_planes, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2),
           512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=6):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
        	stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


cnn = MobileNet()
cnn=cnn.cuda()

#训练过程
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss() #选择误差函数

#保存网络
def saveNet():
    torch.save(cnn, 'toolNet3.pkl')


if __name__ == '__main__':
    accuracyList=[]
    epochlist=[]
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            b_x=Variable(x).cuda()
            b_y=Variable(y).cuda()

            output=cnn(b_x)
            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%200 == 0:

                #saveNet()

                sum_accuracy = 0.0
                num = 0

                for step2,(x2,y2) in enumerate(test_loader):
                    test_x=Variable(x2).cuda()
                    test_y=Variable(y2).cuda()
                    test_output=cnn(test_x)
                    pred_y=torch.max(test_output,1)[1].cuda().data
                    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                    sum_accuracy=sum_accuracy+accuracy
                    num=num+1
                    print('Epoch: ', epoch, '| train loss: %.20f' % loss.item(), '| test accuracy: %.4f' % accuracy)

                print("total accuracy:%.4f"%(sum_accuracy/num))
                accuracyList.append([sum_accuracy/num])
                epochlist.append(epoch)

    plt.plot(epochlist,accuracyList)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()