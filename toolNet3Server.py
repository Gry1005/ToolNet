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
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            #nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            #nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
            #nn.BatchNorm2d(384),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
            #nn.BatchNorm2d(384),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            #nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(3, 2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 6)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out


#加载神经网络

net = torch.load('toolNet3.pkl')
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
    n=0;
    while True:
        conn, addr = sk.accept()
        n=n+1;
        while True:
            #if n%2!=0:
                #object_detect()

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


