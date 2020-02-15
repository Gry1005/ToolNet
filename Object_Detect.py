# encoding:utf-8
import base64
import urllib
import json
import cv2

#获取access_token

host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=pY8H7iqACKeHGamBUcXjofd8&client_secret=xFuqId0ptmZcncB7qa7B2YSNUcGaa8aO'
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
if (content):
    print(content)#<class 'bytes'>
content_str=str(content, encoding="utf-8")
###eval将字符串转换成字典
content_dir = eval(content_str)
access_token = content_dir['access_token']


'''
图像主体检测
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"

# 二进制方式打开图片文件
f = open('D:/BaiduAITest/ObjectDetect/197.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img,"with_face":0}
params = urllib.parse.urlencode(params).encode('utf-8')

request_url = request_url + "?access_token=" + access_token
request1 = urllib.request.Request(url=request_url, data=params)
request1.add_header('Content-Type', 'application/x-www-form-urlencoded')
response = urllib.request.urlopen(request1)
content = response.read()
if content:
    print(content)

string = str(content,'utf-8')
jsonResult = json.loads(string)
#print(jsonResult)

#解析后为int
top=jsonResult['result']['top']
left=jsonResult['result']['left']

width=jsonResult['result']['width']
height=jsonResult['result']['height']

img = cv2.imread('D:/BaiduAITest/ObjectDetect/197.jpg')
cv2.rectangle(img,(left,top),(left+width,top+height),(0,255,0),2)
#cv2.imshow("image",img)
cv2.imwrite('D:/BaiduAITest/ObjectDetect/197'+'-result.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
