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
                name = "-1"
        # update the list of names
        # if name not in names:
        if name != "Unknown":
            names.append(name)
    return names




def detect_and_predict_mask(frame,facenet,masknet):
	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

	print(type(frame))
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
		name="detecteion on progress"
		for(box,pred) in zip(locs,preds):
			(startx,starty,endx,endy)=box
			sub_face = im[starty:endy, startx:endx]

			FaceFileName = "static/test.jpg" #Saving the current image from the webcam for testing.
			
			cv2.imwrite(FaceFileName, sub_face)


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
				flag=1
				# engine=pyttsx3.init()
				# engine.say(res[0]['names'])
				# engine.runAndWait() 
				# time.sleep(30)
				print("#########################################")
			else:
				name=val

	
			(mask,withoutmask)=pred
			label="mask" if mask>withoutmask else "no mask"
			color=(0,255,0) if label=="mask" else (0,0,255)
			l=label
			label="{}: {:.2f}%".format(label,max(mask,withoutmask)*100)
			cv2.rectangle(im,(startx,starty),(endx,endy),color,2)
			cv2.putText(im,name+" "+label,(startx,endy),cv2.FONT_ITALIC,1.0,color,2)
			cv2.imshow("esc",im) 
			key = cv2.waitKey(1)
			engine=pyttsx3.init()
			engine.setProperty("rate",100)
			if name!="detecteion on progress":
				print(label)
				if l=="no mask":
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
					flag=2
					return "failed"
					break

			if flag==1:
				print("sucessssssssssssssssssssssssssssssssssssss")
				if key == 27: #The Esc key
					print(engine.getProperty("voices"))
					flag=2
					return name
					break

				


		if flag==2:
			break



prototxtpath="face_detector/deploy.prototxt"
print(prototxtpath)
weightspath="face_detector/res10_300x300_ssd_iter_140000.caffemodel"
facenet=cv2.dnn.readNet(prototxtpath,weightspath)
masknet=load_model("mask_detector.model")
a=val()