# Import the necessary packages 
from flask import Flask, request
from werkzeug.datastructures import ImmutableMultiDict
import werkzeug.formparser
from waitress import serve

import json
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import imutils 
import dlib
import cv2 
import csv
import numpy as np
import datetime

app = Flask(__name__)
@app.route('/face_detect', methods=['POST'])
def detect():
    
    data = {}
 
    f = request.files['file']
    current_milli_time = round(datetime.datetime.now().timestamp() * 1000)
    f.filename = "{}.jpg".format(current_milli_time)
    f.save("face_detect/"+f.filename)
    # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
    EAR_THRESHOLD = 0.2
    # Declare another costant to hold the consecutive number of frames to consider for a blink 
    CONSECUTIVE_FRAMES = 20 
    # Another constant which will work as a threshold for MAR value
    MAR_THRESHOLD = 14
    
    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Grab the indexes of the facial landamarks for the left and right eye respectively 
    (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    image = cv2.imread("face_detect/"+f.filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    rects = detector(image, 1)

    if len(rects)==1:
        
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(shape)
            # Draw a rectangle over the detected face 
            
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend] 
            mouth = shape[mstart:mend]
            # Compute the EAR for both the eyes 
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            #live datawrite in csv
            
            MAR = mouth_aspect_ratio(mouth)
            # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
            # Thus, count the number of frames for which the eye remains closed 
            if EAR < EAR_THRESHOLD: 
                data['status'] = "ER"
                data['eye'] = False
            else:
                data['status'] = "SR"
                data['eye'] = True
            # Check if the person is yawning
            if MAR > MAR_THRESHOLD:
                data['status'] = "ER"
                data['yawn'] = False
            else:
                data['status'] = "SR"
                data['yawn'] = True
        
    elif len(rects)==0:
        data['status'] = "ER"
        data['message'] = "Face not available"

    else:
        data['status'] = "ER"
        data['message'] = "Multiple face detected"
        
    return (data)
        
if __name__ == "__main__":
    app.debug = True    
    serve(app,host='0.0.0.0',port=5005)    
