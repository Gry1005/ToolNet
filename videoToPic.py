# encoding: utf-8

import cv2

def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                numFrame += 1
                newPath = svPath + str(numFrame) + ".jpg"
                print(newPath)
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        if cv2.waitKey(10) == 27:
            break

#videoPath="D:/dataSet/toolNetData/VID_20181205_232111.mp4"
#savePicturePath="D:/dataSet/toolNetData/ceDianBi/"

videoPath="D:/dataSet/ToolData2/20190512_175343.mp4"
savePicturePath="D:/dataSet/ToolData2/ttbs/"

getFrame(videoPath, savePicturePath)
