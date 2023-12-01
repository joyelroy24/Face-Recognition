from flask import *
from database import *
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from imutils.video import VideoStream
import pickle
import math
import uuid
import face_recognition
import timer
from imutils import paths
import pyttsx3
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def detect_and_predict_mask(frame,facenet,masknet):
	(h, w) = frame.shape[:2]
	blob=cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	facenet.setInput(blob)
	detections=facenet.forward()
	faces=[]
	locs=[]
	preds=[]
	for i in range(0,detections.shape[2]):
		confidence=detections[0,0,i,2]
		if confidence>0.5:
			box=detections[0,0,i,3:7] * np.array([w,h,w,h])
			(startx,starty,endx,endy)=box.astype("int")
			(startx,starty)=(max(0,startx),max(0,starty))
			(endx,endy)=(min(endx,w-1),min(endy,h-1))
			face=frame[starty:endy,startx:endx]
			if face.any():
				face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
				face=cv2.resize(face,(224,224))
				face=img_to_array(face)
				face=preprocess_input(face)
				faces.append(face)
				locs.append((startx,starty,endx,endy))
	if len(faces)>0:
		faces=np.array(faces,dtype="float32")
		preds = masknet.predict(faces, batch_size=32)
	return(locs,preds)

def val():
	size=4
	classifier=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	webcam = cv2.VideoCapture(0)
	l=" "
	(rval,im)=webcam.read()
	flag=0
	vanew="hello"
	while True:
		(rval,im)=webcam.read()
		im=cv2.flip(im,1,0)
		(locs, preds) = detect_and_predict_mask(im, facenet, masknet)
		
		label="detection"
		for(box,pred) in zip(locs,preds):
			(startx,starty,endx,endy)=box
			(mask,withoutmask)=pred
			label="mask" if mask>withoutmask else "no mask"
			color=(0,255,0) if label=="mask" else (0,0,255)
			l=label
			label="{}: {:.2f}%".format(label,max(mask,withoutmask)*100)
			cv2.rectangle(im,(startx,starty),(endx,endy),color,2)
			cv2.putText(im,label,(startx,endy),cv2.FONT_ITALIC,1.0,color,2)
			cv2.imshow("esc",im)
			key=cv2.waitKey(1)&0xff
			if key==ord("q" or "Q"):
				flag=1
				break
		if flag==1:
			break


prototxtpath="face_detector/deploy.prototxt"
print(prototxtpath)
weightspath="face_detector/res10_300x300_ssd_iter_140000.caffemodel"
facenet=cv2.dnn.readNet(prototxtpath,weightspath)
masknet=load_model("mask_detector.model")
val()