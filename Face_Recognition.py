# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:58:56 2021

@author: Daniyar Syrlybayev
"""

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier(r'C:\Users\Daniyar Syrlybayev\Desktop\Python\NumPy\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

people = ["Without Mask", "With Mask"]
features = np.load('features.npy', allow_pickle = True)
labels = np.load('labels.npy', allow_pickle = True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

image = cv.imread("without_mask.jpg")


gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
faces_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

for (x, y, w, h) in faces_rect:
    cv.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0))

faces_roi = gray[y:(y+h), x:(x+w)]
label, confidence = face_recognizer.predict(faces_roi)
cv.putText(image, people[label], (50, 50), cv.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0))
cv.imshow("Image", image)

print(people[label])



cv.waitKey(0)