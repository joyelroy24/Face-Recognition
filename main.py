
from flask import Blueprint,request,render_template,flash,session
import uuid

import os
from core import *


app=Flask(__name__)
app.secret_key="aa"


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/staff')
def staff():
	title=request.args['title']
	return render_template('staff.html',title=title)

@app.route('/markattanadance')
def markattanadance():
	s=val()
	return render_template('markattandance.html',s=s)

@app.route('/login/',methods=['get','post'])
def login():
	if "submits" in request.form:
		users=request.form['username']
		passs=request.form['password']
		q="select * from login where username='%s' and password='%s'" %(users,passs)
		res=select(q)
		if res:
			session['login_id']=res[0]['login_id']
			if res[0]['usertype']=="admin":
				return redirect(url_for('adminhome'))
			
			else:
				flash("Invalid Username and password")
		else:
			flash("Invalid Username and password")


	return render_template('login.html')

@app.route('/adminhome')
def adminhome():
	return render_template('adminhome.html')

@app.route('/adminmanagestaff',methods=['get','post'])
def adminmanagestaff():
	data={}
	if "submits" in request.form:
		name=request.form['name']
		aadhar=request.form['aadhar']
		date=request.form['date']


		i=request.files['image']
		path="static/uploads/"+str(uuid.uuid4())+i.filename
		i.save(path)


		# q="insert into login values(null,'%s','%s','staff')" %(username,password)
		# id=insert(q)
		q="insert into staff values(null,'%s','%s','%s','%s','0')" %(name,aadhar,date,path)
		id=insert(q)

		# path = 'static/uploads/'
		path=""
		# Check whether the   
		# specified path is   
		# an existing file 
		pid=str(id)
		isFile = os.path.isdir("static/trainimages/"+pid)  
		print(isFile)
		if(isFile==False):
			os.mkdir('static\\trainimages\\'+pid)
		image1=request.files['image1']
		path="static/trainimages/"+pid+"/"+str(uuid.uuid4())+image1.filename
		image1.save(path)

		image2=request.files['image2']
		path="static/trainimages/"+pid+"/"+str(uuid.uuid4())+image2.filename
		image2.save(path)

		image3=request.files['image3']
		path="static/trainimages/"+pid+"/"+str(uuid.uuid4())+image3.filename
		image3.save(path)
		enf("static/trainimages/")
		
		flash('Added successfully...')
		return redirect(url_for('adminmanagestaff'))

	q="select * from staff"
	res=select(q)
	data['staffs']=res


	return render_template('adminmanagestaff.html',data=data)


@app.route('/uploadimagesforstaff',methods=['get','post'])
def uploadimagesforstaff():
	data={}
	staff_id=request.args['ids']
	if 'val' in request.args:
		val=request.args['val']
	else:
		val=None

	if val=="1":
		# pritn("Haii")
		import cv2

		# 1.creating a video object
		video = cv2.VideoCapture(0) 
		# 2. Variable
		a = 0
		# 3. While loop
		while True:
		    a = a + 1
		    # 4.Create a frame object
		    check, frame = video.read()
		    # Converting to grayscale
		    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		    # 5.show the frame!
		    cv2.resize(frame,(800,800),interpoloation=cv2.INTER_NEAREST)
		    cv2.imshow("Capturing",frame)
		    # 6.for playing 
		    key = cv2.waitKey(1)
		    if key == ord('q'):
		        break
		# 7. image saving
		showPic = cv2.imwrite("static/trainimages/"+staff_id+"/"+str(uuid.uuid4())+".jpg",frame)
		print(showPic)
		q="update staff set noofinput='1' where staff_id='%s'" %(staff_id)
		update(q)
		# 8. shutdown the camera
		video.release()
		cv2.destroyAllWindows
		return redirect(url_for('adminmanagestaff'))
	if val=="2":
		import cv2

		# 1.creating a video object
		video = cv2.VideoCapture(0) 
		# 2. Variable
		a = 0
		# 3. While loop
		while True:
		    a = a + 1
		    # 4.Create a frame object
		    check, frame = video.read()
		    # Converting to grayscale
		    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		    # 5.show the frame!
		    cv2.imshow("Capturing",frame)
		    # 6.for playing 
		    key = cv2.waitKey(1)
		    if key == ord('q'):
		        break
		# 7. image saving
		showPic = cv2.imwrite("static/trainimages/"+staff_id+"/"+str(uuid.uuid4())+".jpg",frame)
		print(showPic)
		q="update staff set noofinput='2' where staff_id='%s'" %(staff_id)
		update(q)
		# 8. shutdown the camera
		video.release()
		cv2.destroyAllWindows
		return redirect(url_for('adminmanagestaff'))
	if val=="3":
		import cv2

		# 1.creating a video object
		video = cv2.VideoCapture(0) 
		# 2. Variable
		a = 0
		# 3. While loop
		while True:
		    a = a + 1
		    # 4.Create a frame object
		    check, frame = video.read()
		    # Converting to grayscale
		    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		    # 5.show the frame!
		    cv2.imshow("Capturing",frame)
		    # 6.for playing 
		    key = cv2.waitKey(1)
		    if key == ord('q'):
		        break
		# 7. image saving
		showPic = cv2.imwrite("static/trainimages/"+staff_id+"/"+str(uuid.uuid4())+".jpg",frame)
		print(showPic)
		q="update staff set noofinput='3' where staff_id='%s'" %(staff_id)
		update(q)
		# 8. shutdown the camera
		video.release()
		cv2.destroyAllWindows
		return redirect(url_for('adminmanagestaff'))
	q="select noofinput from staff where staff_id='%s'" %(staff_id)
	print(q)
	res=select(q)
	if res:
		data['val']=res[0]['noofinput']
	else:
		data['val']=0
	data['ids']=staff_id
	return render_template('uploadimagesforstaff.html',data=data)



app.run(debug=True,port=5002)