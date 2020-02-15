#encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import socket

import base64
import urllib
import json
import cv2

#主体识别函数:

def object_detect():
    # 获取access_token

    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=pY8H7iqACKeHGamBUcXjofd8&client_secret=xFuqId0ptmZcncB7qa7B2YSNUcGaa8aO'
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    if (content):
        print(content)  # <class 'bytes'>
    content_str = str(content, encoding="utf-8")
    ###eval将字符串转换成字典
    content_dir = eval(content_str)
    access_token = content_dir['access_token']

    '''
    图像主体检测
    '''

    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"

    # 二进制方式打开图片文件
    imgPath='D:/toolTest/0/test.jpg'
    f = open(imgPath, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img, "with_face": 0}
    params = urllib.parse.urlencode(params).encode('utf-8')

    request_url = request_url + "?access_token=" + access_token
    request1 = urllib.request.Request(url=request_url, data=params)
    request1.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request1)
    content = response.read()
    if content:
        print(content)

    string = str(content, 'utf-8')
    jsonResult = json.loads(string)
    # print(jsonResult)

    # 解析后为int
    top = jsonResult['result']['top']
    left = jsonResult['result']['left']

    width = jsonResult['result']['width']
    height = jsonResult['result']['height']

    img = cv2.imread(imgPath)
    cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
    # cv2.imshow("image",img)
    cv2.imwrite('C:/apache-tomcat-9.0.19/webapps/image/result.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])



def loadData():
    #加载测试集
    path2="D:\\toolTest"
    test_data = torchvision.datasets.ImageFolder(path2,transform=transforms.Compose([transforms.Resize((227,227)),transforms.CenterCrop(227),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_loader = Data.DataLoader(test_data,batch_size=1,shuffle=True)
    return test_loader

#神经网络定义
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
    def __init__(self, vgg_name='VGG11'):
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


#加载神经网络

net = torch.load('VGG11.pkl')
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
    n=0
    while True:
        conn, addr = sk.accept()
        n=n+1
        while True:

            if n%2!=0:
                object_detect()

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


