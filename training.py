from flask import Blueprint,request,render_template,flash,session
import uuid

import os
from core import *
import cv2
from database import *
from flask import *
import demjson
import numpy as np
from model_manager import Model
import pickle
import math
import cv2
import uuid

import face_recognition
import argparse
import timer

from imutils import paths
import os
import requests
import io
import json
from database import *
import time
import pyttsx3
import time



def enf(path):
	imgpath=path
	knownNames=[]
	knownEncodings=[]
	for subdir in os.listdir(imgpath):
		print(subdir)
		facedir=os.path.join(imgpath,subdir)
		for facename in os.listdir(facedir):
			faceimgpath=os.path.join(facedir,facename)
			name=subdir
			fimg=cv2.imread(faceimgpath)
			rgb=cv2.cvtColor(fimg,cv2.COLOR_BGR2RGB)
			boxes=face_recognition.face_locations(rgb,model='hog')
			encoding=face_recognition.face_encodings(rgb,boxes)
			print(type(knownNames))
			for enc in encoding:

				knownNames.append(name)
				knownEncodings.append(enc)

	data={'names':knownNames,"encodings":knownEncodings}
	f=open('faces.pickles',"wb")
	f.write(pickle.dumps(data))
	print("finished")


def traning():
	img=cv2.imread("h1.jpg")
	img2=cv2.imread("h2.jpg")



	cv2.waitKey(0)
	file=os.path.isdir("static/trainimages/Hridhik")
	if file==False:
		print("not exist")
		os.mkdir("static/trainimages/Hridhik")
		path1="static/trainimages/Hridhik/h1.jpg"
		cv2.imwrite(path,img)
		path2="static/trainimages/Hridhik/h2.jpg"
		cv2.imwrite(path2,img2)
		path2="static/trainimages/Hridhik/h3.jpg"
		cv2.imwrite(path2,cv2.imread("h3.jpg"))
	else:
		path1="static/trainimages/Hridhik/h1.jpg"
		cv2.imwrite(path1,img)
		path2="static/trainimages/Hridhik/h2.jpg"
		cv2.imwrite(path2,img2)
		path2="static/trainimages/Hridhik/h3.jpg"
		cv2.imwrite(path2,cv2.imread("h3.jpg"))

	enf("static/trainimages/")

traning()