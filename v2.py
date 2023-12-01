import argparse
import cv2
import imutils
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from imutils.video import VideoStream

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from database import *
from flask import *
import demjson
from model_manager import Model
import pickle
import math
import uuid

import face_recognition

import timer

from imutils import paths

import requests
import io

import time
import pyttsx3
import time


def detect_and_predict_mask(frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []
		print(detections.shape[2])
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			# print("((((((((((((())))))))))))))))))")
			# print(i)
			confidence = detections[0, 0, i, 2]
			# print(("+++++++++++++++++++++++++++++++"))
			# print(confidence)

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				if face.any():
					face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					face = cv2.resize(face, (224, 224))
					face = img_to_array(face)
					face = preprocess_input(face)

					# add the face and bounding boxes to their respective
					# lists
					faces.append(face)
					locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their corresponding
		# locations
		return (locs, preds)





def rec_face_image(imagepath):
    print(imagepath)

    data = pickle.loads(open('faces.pickles', "rb").read())

    # load the input image and convert it from BGR to RGB
    image = cv2.imread(imagepath)
    #print(image)

    h,w,ch=image.shape
    print(h)
    print("________________________")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,model='hog')
    print(boxes)
    encodings = face_recognition.face_encodings(rgb, boxes)
    print("************************************")
    print(encodings)
    # initialize the list of names for each face detected
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.4)
        name = "Unknown"
        print(matches)
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            print(list(enumerate(matches)))
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            print("++++++",matchedIdxs)
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                print(name)
                counts[name] = counts.get(name, 0) + 1
            print(counts, " rount ")
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            if len(counts) == 1:
            	print("""""")
            	print(counts)
            	key=counts.get
            	print(key)
            	name = max(counts, key=counts.get)
            	print("-----------------")
            	print(name)
            else:
                name = "NOT DETECTED"
        # update the list of names
        # if name not in names:
        if name != "Unknown":
            names.append(name)
    return names

def val():
	size = 4
	# We load the xml file
	classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

	webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
	# time.sleep(2.0)
	l=" "
	(rval, im) = webcam.read()
	flag=0
	valnew="hello"
	name="not detected"
	x=50
	y=50
	while True:
		
		(rval, im) = webcam.read()
		im=cv2.flip(im,1,0) #Flip to act as a mirror

		# Resize the image to speed up detection

		(locs, preds) = detect_and_predict_mask(im, faceNet, maskNet)
		label="dectection"
		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
			print("maskkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			l=label
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			# cv2.putText(frame, label, (startX, startY - 10),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
			cv2.rectangle(im, (startX, startY), (endX, endY), color, 2)



		mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))
		# detect MultiScale / faces 
		faces = classifier.detectMultiScale(mini)
	
		# Draw rectangles around each face
		flag=0
		for f in faces:
			print("faceeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
			(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
			# cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)

			#Save just the rectangle faces in SubRecFaces
			sub_face = im[y:y+h, x:x+w]

			FaceFileName = "static/test.jpg" #Saving the current image from the webcam for testing.
			
	
			cv2.imwrite(FaceFileName, sub_face)
	
			# break
			print("""""""""""""""""""""""""""""")
			if valnew=="hello":
				val=rec_face_image(FaceFileName)
				print(val)
			# print("user",vals)
				str1=""
				for ele in val:  
					str1 = ele
				print(str1)
				val=str1.replace("'","")
				print("val : ",val)
				q="select *,name as names from staff where staff_id='%s'" %(val)
				res=select(q)
				if res:
					name=res[0]['names']
					valnew=name
					# engine=pyttsx3.init()
					# engine.say(res[0]['names'])
					# engine.runAndWait() 
					# time.sleep(30)
					print("#########################################")
					print(name)
					q="insert into attandance values(null,'%s',curdate(),curtime())" %(val)
					insert(q)
					flag=1
				else:
					name=val

			# cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)	
		# im=imutils.resize(im,width=1200,height=1000)
		if l=="No Mask":	
			cv2.putText(im,name+"  "+label,(x-20,y-20),cv2.FONT_ITALIC, 1.0, (0,0, 255), 2)
		else:
			cv2.putText(im,name+"  "+label,(x-20,y-20),cv2.FONT_ITALIC, 1.0, (0,255, 0), 2)

		cv2.imshow('Capture(Press Esc to exit)',   im)

		key = cv2.waitKey(1)
		engine=pyttsx3.init()
		engine.setProperty("rate",100)
		if name!="not detected":
			print(label)
			if l=="No Mask":
				engine.say(name+"please ware mask")
				print("please ware mask")
				engine.runAndWait() 
			else:
				engine.say(name+"you safe from corona virus by waring"+label+"precentage correctly")
				print("you ware"+label+"precentage correctly")
				engine.runAndWait() 
		else:
			engine.say("you ware"+label+"precentage correctly")
			engine.runAndWait() 

		print("key : ",key)
		# # if Esc key is press then break out of the loop 
		if flag==0:	
			print("failedddddddddddddddddddddddddddddddddddd")
			if key == 27: #The Esc key

				return "failed"
				break

		if flag==1:
			print("sucessssssssssssssssssssssssssssssssssssss")
			if key == 27: #The Esc key
				print(engine.getProperty("voices"))
				return name
				break


app=argparse.ArgumentParser()
app.add_argument("-f", "--face", type=str,default="face_detector",help="path to face detector model directory")
app.add_argument("-m", "--model",type=str,default="mask_detector.model",help="path to trained face mask detector model")
app.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(app.parse_args())
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
print(prototxtPath)
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])
print(args['face'])

# args=vars(app.parse_args())
# print(args["path"])
a=val()



