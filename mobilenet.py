import cv2
from datetime import datetime
from web import *

def init_detection():
    global classNames
    classNames = []
    classFile = '/home/pi/IoT_Counter/coco.names' # you should change the path of this file 
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = '/home/pi/IoT_Counter/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # you should change the path of this file 
    weightsPath = '/home/pi/IoT_Counter/frozen_inference_graph.pb' # you should change the path of this file 

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
    #print(classIds, bbox)
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
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    # cap.set(10,70)
    global data
    
    while True:

        success, img = cap.read() 
        result, objectInfo, confidence , Name = getObjects(img, 0.45, 0.2, objects=['person']) # put the object name what you want to detect in coco names file
        #print(objectInfo)
        print(confidence, end=': ')
        print(Name)
        print(count)
        now = datetime.now()
        t_h = str(now.hour)
        t_m = str(now.minute)
        t_s = str(now.second)
        data.append((t_h + ":" + t_m + ":" + t_s,count))
        cv2.imshow("Output", img)
        cv2.waitKey(1)
