import cv2

import os

Path='D:/pycharmWorkspace/toolNet/dataSet5/test/5'

for filename in os.listdir(Path):              #listdir的参数是文件夹的路径
    print (filename)                               #此时的filename是文件夹中文件的名称

    img = cv2.imread(Path+"/"+filename)

    (h, w) = img.shape[:2] #10
    center = (w // 2, h // 2) #11

    M = cv2.getRotationMatrix2D(center, 180, 1.0) #12
    rotated = cv2.warpAffine(img, M, (w, h))

    #cv2.imshow("image",rotated)
    cv2.imwrite(Path+"/"+"r-"+filename, rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 100])