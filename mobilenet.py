import cv2
from datetime import datetime
from web import *
import os

global opt
opt = 1

def init_detection():
    global classNames
    classNames = []
    classFile = '/home/pi/IoT_Counter/coco.names' # (!) path (!)
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = '/home/pi/IoT_Counter/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # (!) path (!) 
    weightsPath = '/home/pi/IoT_Counter/frozen_inference_graph.pb' # (!) path (!)

    global net
    global confidence
    global Name
    global count
    
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5, 127.5))
    net.setInputSwapRB(True)

    confidence = 0
    Name = ''
    count = 0
    
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold = thres, nmsThreshold = nms) 
    global classNames
    if len(objects) == 0:
        objects = classNames
    objectInfo = []

    if len(classIds) != 0:
        global confidence
        global Name
        global count
        count = 0
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box, className]) 
                cv2.rectangle(img, box, color = (0,255,0), thickness=2)
                cv2.putText(img, classNames[classId-1].upper(), (box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img, str(round(confidence*100,2)), (box[0]+200,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                Name = className
                count += 1
        
    return img, objectInfo, confidence, Name
        
def counting():

    cap = cv2.VideoCapture(0) # rasp-camera
    # cap = cv2.VideoCapture('http://192.168.0.16:8090/?action=stream')
    cap.set(3,640)
    cap.set(4,480)
    # cap.set(10,70)
    
    global data
    global opt
    
    d_cnt = 0
    sum_cnt = 0
    
    while True:

        success, img = cap.read() 
        result, objectInfo, confidence , Name = getObjects(img, 0.45, 0.2, objects=['person'])
        print(Name,end=" ")
        print(str(confidence) + "%")
        print("count:"+str(count))
        now = datetime.now()
        t_h = str(now.hour)
        t_m = str(now.minute)
        t_s = str(now.second)
        
        print("opt:" + str(opt))
        if opt==1:
            data.append((t_h + ":" + t_m + ":" + t_s,count)) # data
        elif opt==2:
            cur_t = t_h + ":" + t_m
            d_cnt += 1
            sum_cnt += count
            if data[-1][0]!=cur_t :
                data.append((cur_t,sum_cnt/d_cnt)) # data
                d_cnt = 0
                sum_cnt = 0
        
        if os.path.isfile("static/image/camimg.jpg"):
            os.remove("static/image/camimg.jpg")
        cv2.imwrite("static/image/camimg.jpg", img)
        cv2.imshow("Output", img)
        cv2.waitKey(1)
