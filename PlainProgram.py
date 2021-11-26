# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 02:22:21 2021

@author: Daniyar Syrlybayev
"""

import cv2 as cv
import numpy as np
import time
import serial

#serial_comm = serial.Serial('COM3', 9600)
#serial_comm.timeout = 1

def GetLargestContour(CannyImage, HostImage, threshold):
    contours, heirarchy = cv.findContours(CannyImage, 
                                          cv.RETR_LIST, 
                                          cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse= True)
    if cv.contourArea(sorted_contours[0]) > threshold:
        cv.drawContours(HostImage, sorted_contours[0], -1, [0, 0, 255], 
                        thickness = 3)
        return sorted_contours[0]
    else:
        cv.putText(HostImage, "Put your phone closer", (10, 
        HostImage.shape[0]//2), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        return []
    

def GetContourCoordinates(contour):
    maxi = []
    mini = []
    maxsum = 0
    minsum = 10000
    for i in contour:
        suma = i[0][0] + i[0][1]
        if suma > maxsum:
            maxsum = suma
            maxi = i[0]
        if suma < minsum :
            minsum = suma
            mini = i[0]
    corner1 = (mini[0], mini[1])
    corner2 = (maxi[0], maxi[1])
    return (corner1, corner2)

def CheckColor(img, COLOR):
    if COLOR == "Green":
        Low = np.array([25, 72, 52])
        High = np.array([102, 255, 255])
    elif COLOR == "Blue":
        Low = np.array([110,50,50])
        High = np.array([130,255, 255])
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    
    Mask = cv.inRange(hsv, Low, High)
    result = cv.bitwise_and(img, img, mask = Mask)
    return result


def DetectPositiveStatus(PhoneImage, COLOR):
    cv.imshow("Phone Image", PhoneImage)
    Result = CheckColor(PhoneImage, COLOR)
    Canny = cv.Canny(Result, 125, 175)
    cv.imshow(COLOR, Canny)
    Box = GetLargestContour(Canny, PhoneImage, 2000)
    Coordinates = GetContourCoordinates(Box)
    BoxImage = PhoneImage[(Coordinates[0][1]):(Coordinates[1][1]), 
                           (Coordinates[0][0]):(Coordinates[1][0])]
    cv.imshow("Box", BoxImage)
    return BoxImage


def CheckStatus():
    StatusFlag = False
    _, image = video.read()
    BluredImage = cv.medianBlur(image, 5)
    CannyImage = cv.Canny(BluredImage, 125, 175)
    result = GetLargestContour(CannyImage, image, 10000)
    
    if len(result) > 0:
        coordinates = GetContourCoordinates(result)
        PhoneImage = image[(coordinates[0][1]-10):(coordinates[1][1]+10), 
                           (coordinates[0][0]-10):(coordinates[1][0]+10)]
        try:
            Box = DetectPositiveStatus(PhoneImage, "Green")
            if Box.shape[0] * Box.shape[1] >= 5000:
                StatusFlag = True
                cv.putText(image, "Second Stage Check", (image.shape[1]//3, 
                                                         image.shape[0]//2),
                cv.FONT_HERSHEY_DUPLEX, 1, (0, 225, 0))
        except:
            pass
    cv.imshow("ResultFinal", image)
    return StatusFlag


face_cascade = cv.CascadeClassifier(r'C:\Users\Daniyar Syrlybayev\Desktop\Python\NumPy\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

people = ["Without Mask", "With Mask"]
features = np.load('features.npy', allow_pickle = True)
labels = np.load('labels.npy', allow_pickle = True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


def CheckMask():
    StatusFlag = False
    _, image = video.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("Image w/t rec", image)
    
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, 
                                               minNeighbors = 3)
    if len(faces_rect) == 1:
        for (x, y, w, h) in faces_rect:
            cv.rectangle(image, (x-15, y-15), (x+w+15, y+h+15), (0, 255, 0))
        
        faces_roi = gray[y-15:(y+h)+15, x-15:(x+w)+15]
        label, confidence = face_recognizer.predict(faces_roi)
        
        if people[label] == "With Mask":
                StatusFlag = True
                cv.putText(image, "Welcome", (50, 50), cv.FONT_HERSHEY_DUPLEX, 
                           1.1, (0, 255, 0))
        else: 
            cv.putText(image, 'Please, put on your facemask', (50, 50), 
                       cv.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255))
        
        cv.imshow("Image", image)
        return StatusFlag

    
video = cv.VideoCapture(0)
video.set(10, 100)
video.set(3, 500)
video.set(4, 500)


while True:
    AshyqStatus = False
    MaskStatus = False
    counter = 0
    timing = 0
    
    while AshyqStatus == False:
        AshyqStatus = CheckStatus()
        cv.waitKey(1000//30)
    cv.destroyAllWindows()
    
    while counter != 2:
        MaskStatus = CheckMask()
        if MaskStatus == True:
            counter = counter + 1
            time.sleep(1)
        else: 
            counter = 0
        cv.waitKey(1000//30)
    cv.destroyAllWindows()
    
    if (AshyqStatus == True) and (MaskStatus == True):
        Answer = "yes"
    else:
        Answer = "no"
        
    serial_comm.write(Answer.encode())
    time.sleep(6)
    
    print("OK")
    if cv.waitKey(1000//30) & 0xff == ord('q'):
        break

serial_comm.close()


