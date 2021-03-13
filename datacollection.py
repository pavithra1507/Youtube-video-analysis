#!/usr/bin/env python
# python -m pip uninstall pytube pytube3 pytubex
# python -m pip install git+https://github.com/nficano/pytube
import os
import cv2
import numpy as np
import dlib
import time
import pathlib
import sys
#sys.path.append('/path/to/ffmpeg')
#ffmpeg_path = "/s/chopin/l/grad/pavi1998/.local/lib/python3.6/site-packages"
#os.environ["PATH"] += os.pathsep + ffmpeg_path

import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from apiclient.discovery import build
from googleapiclient.discovery import build
from pytube import YouTube
import csv
import subprocess
#from mtcnn.mtcnn import MTCNN

from imutils import face_utils
from scipy.spatial import distance as dist
from os.path import dirname, join
import matplotlib.pyplot as plt

#import ffmpeg
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math

# import cv2
# import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks,draw_marks

# data_dir = "/home/pavithra/Project/frames_ffmpeg/21_input"
# save_dir = "/home/pavithra/Project/frames_ffmpeg/21_pose"
detector = dlib.get_frontal_face_detector()
eye_cascade = cv2.CascadeClassifier('/s/chopin/l/grad/pavi1998/Downloads/project/dlib/python_examples/haarcascade_eye.xml')


# detector = dlib.get_frontal_face_detector()
# win = dlib.image_window()
predictor = dlib.shape_predictor("/s/chopin/l/grad/pavi1998/Downloads/project/dlib/python_examples/shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68


# 获取最大的人脸
def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]

    # print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index


# 从dlib的检测结果抽取姿态估计需要的点坐标
def get_image_points_from_landmark_shape(landmark_shape):
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None

    #2D image points. If you change the image, you need to change vector

    image_points = np.array([
                                (landmark_shape.part(17).x, landmark_shape.part(17).y),  #17 left brow left corner
                                (landmark_shape.part(21).x, landmark_shape.part(21).y),  #21 left brow right corner
                                (landmark_shape.part(22).x, landmark_shape.part(22).y),  #22 right brow left corner
                                (landmark_shape.part(26).x, landmark_shape.part(26).y),  #26 right brow right corner
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),  #36 left eye left corner
                                (landmark_shape.part(39).x, landmark_shape.part(39).y),  #39 left eye right corner
                                (landmark_shape.part(42).x, landmark_shape.part(42).y),  #42 right eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),  #45 right eye right corner
                                (landmark_shape.part(31).x, landmark_shape.part(31).y),  #31 nose left corner
                                (landmark_shape.part(35).x, landmark_shape.part(35).y),  #35 nose right corner
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),  #48 mouth left corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y),  #54 mouth right corner
                                (landmark_shape.part(57).x, landmark_shape.part(57).y),  #57 mouth central bottom corner
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),  #8 chin corner
                            ], dtype="double")
    return 0, image_points

# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img):

    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector( img, 0 )

    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)
# 获取旋转向量和平移向量
def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
                                (6.825897, 6.760612, 4.402142),     #33 left brow left corner
                                (1.330353, 7.122144, 6.903745),     #29 left brow right corner
                                (-1.330353, 7.122144, 6.903745),    #34 right brow left corner
                                (-6.825897, 6.760612, 4.402142),    #38 right brow right corner
                                (5.311432, 5.485328, 3.987654),     #13 left eye left corner
                                (1.789930, 5.393625, 4.413414),     #17 left eye right corner
                                (-1.789930, 5.393625, 4.413414),    #25 right eye left corner
                                (-5.311432, 5.485328, 3.987654),    #21 right eye right corner
                                (2.005628, 1.409845, 6.165652),     #55 nose left corner
                                (-2.005628, 1.409845, 6.165652),    #49 nose right corner
                                (2.774015, -2.080775, 5.048531),    #43 mouth left corner
                                (-2.774015, -2.080775, 5.048531),   #39 mouth right corner
                                (0.000000, -3.116408, 6.097667),    #45 mouth central bottom corner
                                (0.000000, -7.415691, 4.070434)     #6 chin corner
                            ])
    # Camera internalssudo apt-get remove --purge ffmpeg
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000],dtype= "double") # Assuming no lens distortion
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )

    # print("Rotation Vector:\n {}".format(rotation_vector))
    # print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs
     

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)

    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    
    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    # 单位转换：将弧度转换为度
    pitch_degree = int((pitch/math.pi)*180)
    yaw_degree = int((yaw/math.pi)*180)
    roll_degree = int((roll/math.pi)*180)

    return 0, pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree


def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None

        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Pitch:{}, Yaw:{}, Roll:{}'.format(pitch, yaw, roll)
        # print(euler_angle_str)
        return 0, pitch, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None

def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    # print("left",left)
    if left == right and left != 0:
        text = ''
        if left == 1:
            # print('Looking left')
            text = 'Looking left'
        elif left == 2:
            # print('Looking right')
            text = 'Looking right'
        elif left == 3:
            # print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # cv2.putText(img, text, (30, 30), font,  
        #            1, (0, 255, 255), 2, cv2.LINE_AA) 
    elif left==0 or left==None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print("Looking Straight") 
        # cv2.putText(img, 'Looking Straight', (30, 30), font,  
        #            1, (0, 255, 255), 2, cv2.LINE_AA)

# if __name__ == '__main__':
# image_names = os.listdir(data_dir)
count=1
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
# cap = cv2.VideoCapture(0)

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

font = cv2.FONT_HERSHEY_SIMPLEX
face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
font = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((9, 9), np.uint8)
api_key= "AIzaSyCwvpxU1UtpUKzKVte2YBJrhvbtcDp1B5k"
youtube=build('youtube','v3',developerKey=api_key)
nextPageToken=None
total=0
serial=1
output=[]
resultlink=[]
while(total<10):
    req=youtube.search().list(q='makeup voice over video',part='snippet',type='video',videoLicense='creativeCommon',order='relevance',maxResults=50,pageToken=nextPageToken)
    res=req.execute()
    if(res['nextPageToken']==None):
        break;
    else:
        nextPageToken = res['nextPageToken']

        

    
    for item in res['items']:
        video_id=item['id']['videoId']
        output.append(video_id)
    for i in range(len(output)):
        resultlink.append("https://www.youtube.com/watch?v=%s"%(output[i]))
        print(serial,"https://www.youtube.com/watch?v=%s"%(output[i]))
        serial=serial+1
    total=total+1
print("Youtube links obtained!!")
print("number of youtube links",len(resultlink))
for i in range(len(resultlink)):  
    if(resultlink[i]!=None):
        SAVE_PATH = "/run/media/pavi1998/Seagate Portable Drive/Seagate/videos" 

        link=[resultlink[i]] 
        # print(link)
        for i in link:  
            try:  
                # print("i",i)
                yt = YouTube(i)  
            except:  
                print("Connection Error")  
            stream = yt.streams.first()
            # mp4files = yt.filter('mp4')  
            # stream = yt.get(mp4files[-1].extension,mp4files[-1].resolution)    
            try:  
                stream.download(SAVE_PATH)
                print("Downloaded:",i)
            except:  
                print("Some Error!")  
    else:
        break
print('Videos downloaded !!')  

#split into frames

videos=os.listdir("/run/media/pavi1998/Seagate Portable Drive/Seagate/videos")
count="1"
for i in videos:
    directory = count
    parent_dir = "/run/media/pavi1998/Seagate Portable Drive/Seagate/main_input/"
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    
    parent_dir1 = "/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/"
    path1 = os.path.join(parent_dir1, directory) 
    os.mkdir(path1) 
    # print("Directory '%s' created" %directory)
    count=int(count)+1
    count=str(count)

    os.chdir('/run/media/pavi1998/Seagate Portable Drive/Seagate/main_input/%s'%directory)
    subprocess.call(['ffmpeg', '-i', '/run/media/pavi1998/Seagate Portable Drive/Seagate/videos/%s'%(i),'-vf','fps=20','img%d.png'])


global emotion
def nothing(x):
    pass
# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)
    # Read Image
c=0
iteration=1
files=os.listdir("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_input")
# print("folders",(files))
print("Folder creation Task Completed !!")
for folder in files:
    images=os.listdir("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_input/%s"%(folder))
    os.chdir("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_input/%s"%(folder))
    for j in images:
        # print("path",pathlib.Path(j).parent.absolute())

        print("Loading","folder:",folder,"images:",j)
        img=cv2.imread(j)

        # if os.path.isfile(imgpath): 
        # print("Processing file: {}".format(img))
        # img = dlib.load_rgb_image(imgpath)
        # img=cv2.imread(imgpath)
        output = img.copy()
        out=img
        # draw=img
        # print(img)
        dets = detector(img, 1)
        if not dets:
            continue
        

    
        det, scores, idx = detector.run(img, 1, -1)
        # print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            # img = dlib.load_rgb_image(sys.argv[1
            confidencepert=scores[i]
            # print("Confidence score: {}".format(scores[i]))
            if(scores[i]>0.75):
            # win.clear_overlay()
            # win.set_image(img)
            # win.add_overlay(dets)
        # dlib.hit_enter_to_continue()



        # image_names="/home/pavithra/Project/frames_ffmpeg/4_fullinput/frame835.jpg"
        # imgpath=image_names
        # for index, image_name in enumerate(image_names):
        # print("Image:", image_names)
        # imgpath = data_dir +'\\'+ image_name
                facecasc = cv2.CascadeClassifier('/s/chopin/l/grad/pavi1998/Downloads/project/dlib/python_examples/haarcascade_frontalface_default.xml')
                gray_emot = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print("gray",gray_emot)
                faces = facecasc.detectMultiScale(gray_emot,scaleFactor=1.1, minNeighbors=5)
                thresh = img.copy()
                size = img.shape

                if size[0] > 700:
                    h = size[0] / 3
                    w = size[1] / 3
                    img = cv2.resize( img, (int( w ), int( h )), interpolation=cv2.INTER_CUBIC )
                    size = img.shape

                ret, image_points = get_image_points(img)
                if ret != 0:
                    print('get_image_points failed')
                    continue

                ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)
                if ret != True:
                    print('get_pose_estimation failed')
                    continue
                    
                ret, pitch, yaw, roll,pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)

                
                # Yaw:
                if yaw_degree < 0:
                    output_yaw = "face turns left:"+str(abs(yaw_degree))+" degrees"
                    # cv2.putText(draw,output_yaw,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_yaw)
                if yaw_degree == 0:
                    print("face doesn't turns left or right")
                if yaw_degree > 0:
                    output_yaw = "face turns right:"+str(abs(yaw_degree))+" degrees"
                    # cv2.putText(draw,output_yaw,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_yaw)
                # Pitch:
                if pitch_degree > 0:
                    output_pitch = "face downwards:"+str(abs(pitch_degree))+" degrees"
                    # cv2.putText(draw,output_pitch,(20,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_pitch)
                if pitch_degree == 0:
                    print("face not downwards or upwards")
                if pitch_degree < 0:
                    output_pitch = "face upwards:"+str(abs(pitch_degree))+" degrees"
                    # cv2.putText(draw,output_pitch,(20,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_pitch)
                # Roll:
                if roll_degree < 0:
                    output_roll = "face bends to the right:"+str(abs(roll_degree))+" degrees"
                    # cv2.putText(draw,output_roll,(20,120),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_roll)
                if roll_degree == 0:
                    print("face doesn't bend to the right or the left.")
                if roll_degree > 0:
                    output_roll = "face bends to the left:"+str(abs(roll_degree))+" degrees"
                    # cv2.putText(draw,output_roll,(20,120),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    # print(output_roll)
                # Initial status:
                if abs(yaw) < 0.00001 and abs(pitch) < 0.00001 and abs(roll) < 0.00001:
                    # cv2.putText(draw,"Initial ststus",(20,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0))
                    print("Initial ststus")
                # cv2.imshow('img',draw)
                # print("yaw_degree",yaw_degree,"pitch_degree",pitch_degree,"roll_degree",roll_degree)
                if(-21<yaw_degree and yaw_degree<21 and -20<pitch_degree and pitch_degree<20 and -15<roll_degree and roll_degree<15):
                    # print("FACE ACCEPTED")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(gray,1.5,7)
                    # print(eyes)
                    # print("eyes",len(eyes))
                    print("eyes",eyes)
                    print("filename",folder,j)
                    #if (len(eyes)==1 and eyes[0][0]<300 and 50<eyes[0][1] and eyes[0][1]<70 and 30<eyes[0][2] and eyes[0][2]<100 and 30<eyes[1][2] and eyes[1][2]<100 and 30<eyes[1][1] and eyes[1][1]<100):
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                        roi_gray = gray_emot[y:y + h, x:x + w]
                            # print(x,y,w,h)
                            # print(roi_gray)/s/chopin/l/grad/pavi1998/Downloads/project
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                        prediction = model.predict(cropped_img)
                        maxindex = int(np.argmax(prediction))
                        emotion=emotion_dict[maxindex]
                            # cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        # rects = find_faces(img, face_model)
                        # if rects:
                        #     for rect in rects:
                        #         shape = detect_marks(img, landmark_model, rect)
                        #         if(c==0):
                        #             draw_marks(img, shape)
                        #             # cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
                        #             #         1, (0, 255, 255), 2)
                        #             # cv2.imshow("Output", img)
                        #             if cv2.waitKey(1) & 0xFF == ord('r'):
                        #                 for i in range(100):
                        #                     for i, (p1, p2) in enumerate(outer_points):
                        #                         d_outer[i] += shape[p2][1] - shape[p1][1]
                        #                     for i, (p1, p2) in enumerate(inner_points):
                        #                         d_inner[i] += shape[p2][1] - shape[p1][1]
                        #                 break
                        #             # cv2.destroyAllWindows()
                        #             d_outer[:] = [x / 100 for x in d_outer]
                        #             d_inner[:] = [x / 100 for x in d_inner]
                        #             c=1
                        #         cnt_outer = 0
                        #         cnt_inner = 0
                        #         draw_marks(img, shape[48:])
                        #         for i, (p1, p2) in enumerate(outer_points):
                        #             if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        #                 cnt_outer += 1 
                        #         for i, (p1, p2) in enumerate(inner_points):
                        #             if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                        #                 cnt_inner += 1
                        #         # print("cnt_outer",cnt_outer)
                        #         # print("cnt_inner",cnt_inner)
                        #         if cnt_outer > 3 and cnt_inner > 2:
                        #             print('Mouth open')
                        #             cv2.putText(img, 'Mouth open', (30, 30), font,
                        #                     1, (0, 255, 255), 2)
                        #             count=count+1
                        #             continue
                        #         else:
                        #             print('Mouth Closed')
                        #             cv2.putText(img, 'Mouth Closed', (30, 30), font,
                        #                     1, (0, 255, 255), 2)
                        #             cv2.imwrite(os.path.join("/home/pavithra/Project/frames_ffmpeg/21_mouth" , 'frame%d.jpg'% count), out)
                        #             print("going")
                        
                        if(len(eyes)>1 and len(eyes)<3 and 100<eyes[0][0] and eyes[0][0]<300 and 40<eyes[0][2] and eyes[0][2]<80 and 49<eyes[0][1] and eyes[0][1]<95 and 120<eyes[1][0] and eyes[1][0]<400):
                            # a = "Eye Open"
                            # print("inside")
                            if(iteration==1):
                                with open("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s.csv"%(folder), "w", newline='') as csvfile:
                                                    csvwriter = csv.writer(csvfile)  
                                                    filename=folder
                                                    result = [[str(filename), str(confidencepert),emotion]]
                                                    csvwriter.writerows(result)
                                                    iteration=iteration+1
                            else:
                                with open("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s.csv"%(folder) ,"a", newline='') as csvfile:  
                                # creating a csv writer object  
                                                        csvwriter = csv.writer(csvfile)  
                                                        filename=folder
                                                        result = [[str(filename), str(confidencepert),emotion]]
                                                        csvwriter.writerows(result)
                                                        # time_stamp=(count+1)/fps
                                                        # filename='frame%d.png'%count
                                                        # writing the fields  
                                                        # csvwriter.writerow(['Filename','Confidence value','Pose angle','Time stamp']) 
                                                        # for i in range(4):

                                                        # result = [[str(filename), str(confidencepert),emotion]]
                                                        # writing the data rows  
                                                        # csvwriter.writerows(result)
                            if(emotion=="Neutral" or emotion=="Angry" or emotion=="Fearful" or emotion=="Happy"):
                                cv2.imwrite(os.path.join("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s"%(folder) , '%s'%(j)),output)                     
            
                            # cv2.imwrite(os.path.join("/home/pavithra/MS_Project/frames_ffmpeg/30_output" , 'frame%d.png'% count), out)
                        # if(len(eyes)==2 and 200<eyes[1][0] and 50<eyes[1][1] and eyes[1][1]<70):
                            # cv2.imwrite(os.path.join("/home/pavithra/Project/frames_ffmpeg/7_pose" , 'frame%d.jpg'% count), draw)        
                        elif(len(eyes)>=3 and 40<eyes[0][2] and eyes[0][2]<80 and 120<eyes[1][0] and eyes[1][0]<400 and eyes[2][0]>150 and eyes[2][1]>50 and 49<eyes[0][1] and eyes[0][1]<95 and 100<eyes[0][0] and eyes[0][0]<300):
                            if(iteration==1):
                                with open("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s.csv"%(folder) , "w", newline='') as csvfile:
                                                    csvwriter = csv.writer(csvfile)  
                                                    filename=folder
                                                    result = [[str(filename), str(confidencepert),emotion]]
                                                    csvwriter.writerows(result)
                                                    iteration=iteration+1
                            else:
                                with open("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s.csv"%(folder) , "a", newline='') as csvfile:  
                                # creating a csv writer object  
                                                        csvwriter = csv.writer(csvfile)  
                                                        # time_stamp=(count+1)/fps
                                                        # filename='frame%d.png'%count
                                                        # writing the fields  
                                                        # csvwriter.writerow(['Filename','Confidence value','Pose angle','Time stamp']) 
                                                        # for i in range(4):

                                                        # result = [[str(filename), str(confidencepert),emotion]]
                                                        # writing the data rows  
                                                        # csvwriter.writerows(result) 
                                                        filename=folder
                                                        result = [[str(filename), str(confidencepert),emotion]]
                                                        csvwriter.writerows(result)
                            if(emotion=="Neutral" or emotion=="Angry" or emotion=="Fearful" or emotion=="Happy"):

                                cv2.imwrite(os.path.join("/run/media/pavi1998/Seagate Portable Drive/Seagate/main_output/%s"%(folder) , '%s'%(j)),output)  
                            # cv2.imwrite(os.path.join("/home/pavithra/MS_Project/frames_ffmpeg/30_output" , 'frame%d.png'% count), out)
                        # count=count+1
                        # continue
                            
                # if (blink==True):
                    # blink=False
                
                # cv2.putText(img,a,(10,30), font, 1,(0,0,255),2,cv2.LINE_AA)
                
                # for (ex,ey,ew,eh) in eyes:
                #     #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                #     roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
                #     roi_color2 = img[ey:ey+eh, ex:ex+ew]
                #     blur = cv2.GaussianBlur(roi_gray2,(5,5),10)
                #     erosion = cv2.erode(blur,kernel,iterations = 2)
                #     ret3,th3 = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                #     circles = cv2.HoughCircles(erosion,cv2.HOUGH_GRADIENT,4,200,param1=20,param2=150,minRadius=0,maxRadius=0)
                #     try:
                #         for i in circles[0,:]:
                #             if(i[2]>0 and i[2]<55):
                #                 cv2.circle(roi_color2,(i[0],i[1]),i[2],(0,0,255),1)
                #                 cv2.putText(img,"Pupil Pos:",(450,30), font, 1,(0,0,255),2,cv2.LINE_AA)
                #                 cv2.putText(img,"X "+str(int(i[0]))+" Y "+str(int(i[1])),(430,60), font, 1,(0,0,255),2,cv2.LINE_AA)
                #                 d = (i[2]/2.0)
                #                 dmm = 1/(25.4/d)
                #                 diameter.append(dmm)
                #                 cv2.putText(img,str('{0:.2f}'.format(dmm))+"mm",(10,60), font, 1,(0,0,255),2,cv2.LINE_AA)
                #                 cv2.circle(roi_color2,(i[0],i[1]),2,(0,0,255),3)
                #                 #cv2.imshow('erosion',erosion)
                #     except Exception as e:
                #         pass

                # rects = find_faces(img, face_model)

                # for rect in rects:
                #     shape = detect_marks(img, landmark_model, rect)
                #     mask = np.zeros(img.shape[:2], dtype=np.uint8)
                #     mask, end_points_left = eye_on_mask(mask, left, shape)
                #     mask, end_points_right = eye_on_mask(mask, right, shape)
                #     mask = cv2.dilate(mask, kernel, 5)
                    
                #     eyes = cv2.bitwise_and(img, img, mask=mask)
                #     mask = (eyes == [0, 0, 0]).all(axis=2)
                #     eyes[mask] = [255, 255, 255]
                #     mid = (shape[42][0] + shape[39][0]) // 2
                #     eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                #     threshold = cv2.getTrackbarPos('threshold', 'image')
                #     _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                #     thresh = process_thresh(thresh)
                    
                #     eyeball_pos_left = contouring(thresh[:, 0:int(mid)], int(mid), img, end_points_left)
                #     eyeball_pos_right = contouring(thresh[:, int(mid):], int(mid), img, end_points_right, True)
                #     print(eyeball_pos_left)
                #     print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
                #     if(eyeball_pos_left==0 or eyeball_pos_left==None):
                # cv2.imwrite(os.path.join("/home/pavithra/Project/frames_ffmpeg/24_filter" , 'frame%d.jpg'% count), draw)
                    # for (x, y) in shape[36:48]:
                    #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                                
                        # cv2.imshow('eyes', img)
                        # else:
                        #     c=c+1
                        #     continue
                                        # cv2.imwrite(save_dir+"\\"+os.path.splitext(imgpath)[0]+'_pose_estimate.jpg',draw)
                        # count=count+1
                        # continue
                    # count=count+1
                    #else:
                        #count=count+1
                        #continue
                else:
                    # count=count+1
                    continue
        # else:
        #     break
    print("Directory completed:",folder)
    os.chdir('..')

