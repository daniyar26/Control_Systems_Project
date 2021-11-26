# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:30:05 2021

@author: Daniyar Syrlybayev
"""

import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(r'C:\Users\Daniyar Syrlybayev\Desktop\Python\NumPy\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

people = ["Without Mask", "With Mask"]
features = np.load('features.npy', allow_pickle = True)
labels = np.load('labels.npy', allow_pickle = True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


video = cv.VideoCapture(0)
video.set(10, 100)
video.set(3, 500)
video.set(4, 500)

while True:
    _, image = video.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("Image w/t rec", image)
    
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
    if len(faces_rect) == 1:
        for (x, y, w, h) in faces_rect:
            cv.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0))
        
        faces_roi = gray[y-10:(y+h)+10, x-10:(x+w)+10]
        label, confidence = face_recognizer.predict(faces_roi)
        
        if people[label] == "With Mask":
            cv.putText(image, people[label], (50, 50), cv.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0))
        else: 
            cv.putText(image, people[label], (50, 50), cv.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255))
        
        cv.imshow("Image", image)
        
    if cv.waitKey(1000//30) & 0xFF == ord("q") :
        break