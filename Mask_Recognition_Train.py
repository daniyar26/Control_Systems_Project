# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 12:14:44 2021

@author: Daniyar Syrlybayev
"""

import os
import numpy as np
import cv2 as cv



face_cascade = cv.CascadeClassifier(r'C:\Users\Daniyar Syrlybayev\Desktop\Python\NumPy\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
people = []

for i in os.listdir(r'C:\Users\Daniyar Syrlybayev\Desktop\Shape_Recognition\FaceDetection\dataset'):
    people.append(i)

DIR = r"C:\Users\Daniyar Syrlybayev\Desktop\Shape_Recognition\FaceDetection\dataset"

features = [] #image arrays of the faces
labels = [] #corresponding label

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            
            try:
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            except:
                continue
            
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
            
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:(y+h), x:(x+w)]    
            
            features.append(faces_roi)
            labels.append(label)
            
            
create_train()

print("Train in created------------")

features = np.array(features, dtype = "object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer on the Features list and labels
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)











