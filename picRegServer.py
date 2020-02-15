#encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import socket

def loadData():
    #加载测试集
    path2="D:\\toolTest"
    test_data = torchvision.datasets.ImageFolder(path2,transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]))
    test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=True)
    return test_loader

#神经网络定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #一般来说一个大卷积层包括卷积层，激活函数和池化层
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3, #表示原始图片有多少层，也就是有多少不同种类的特征值，如RGB图片，有红，绿，蓝三个值
                out_channels=64,#表示输出多少个不同种类的特征值；也就是对同一个图片块，有16个过滤器同时工作
                kernel_size=3, #一个过滤器的长和宽都是五个像素点
                stride=1, #相邻两次扫描的图片块之间相隔几个像素点
                padding=1, #在图片周围多出2圈0值，防止过滤器的某一边超过图片边界，如何计算：if stride=1,padding=(kernel_size-1)/2，保证提取出的新图片长宽和原图一样
            ),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #池化层向下筛选需要的部分
            nn.MaxPool2d(
                kernel_size=2, #使用一个长宽为2的池化过滤器
            ),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,3,1,1), #输入的图片有16层，输出图片有32层
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # 输入的图片有16层，输出图片有32层
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  # 输入的图片有16层，输出图片有32层
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),  # 输入的图片有16层，输出图片有32层
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1=nn.Linear(512*7*7,128) #输入的高度是512，长宽为7，因为经过5次池化
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6) #输出6个值

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x) #x中的数据有四个维度:(batch,32,7,7)
        x=self.conv3(x)
        x=self.conv4(x)
        x = self.conv5(x)
        x=x.view(x.size(0),-1) #保留batch,数据变为二维：(batch,32*7*7);因为输出层只接受一维数据作为输入
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output



#加载神经网络

net = torch.load('LeNetNormCuda.pkl')
net = net.cuda()

def testPic():
    #使用神经网络进行测试单张图片的结果
    test_loader=loadData()
    for step2, (x2, y2) in enumerate(test_loader):
        test_x = Variable(x2).cuda()
    #   test_y = Variable(y2)
        test_output = net(test_x)
        pred_y = torch.max(test_output, 1)[1].data
    #   print(pred_y.cpu().numpy()[0])
        return pred_y.cpu().numpy()[0],test_output[0].data.cpu().numpy()

#0 刻刀，1 钳子，2 一字起，3 测电笔，4 套筒扳手， 5 水平仪

if __name__ == '__main__':
    #result=testPic()
    #print(result)

    sk = socket.socket()
    sk.bind(("localhost", 8888))
    sk.listen(5)
    print("server started!")
    while True:
        conn, addr = sk.accept()
        while True:
            accept_data = str(conn.recv(1024),encoding="utf8")
            print("".join(["接收内容：", accept_data, "     客户端口：", str(addr[1])]))

            result = testPic()
            print("图片测试结果：" +str(result[0]))

            # 0 刻刀，1 钳子，2 一字起，3 测电笔，4 套筒扳手， 5 水平仪

            send_data = str(result[0])+":"+str(result[1][0])+"|"+str(result[1][1])+"|"+str(result[1][2])+"|"+str(result[1][3])+"|"+str(result[1][4])+"|"+str(result[1][5])
            print("发送数据："+send_data)

            conn.sendall(bytes(send_data, encoding="utf8"))

            break;

        conn.close()  # 跳出循环时结束通讯


