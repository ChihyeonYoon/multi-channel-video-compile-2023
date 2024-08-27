import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
import torch.backends.cudnn as cudnn

import imutils
from imutils import face_utils
import dlib
dlib.DLIB_USE_CUDA = True

import cv2
import face_recognition

import mediapipe

from glob import glob
import os

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = dlib.get_frontal_face_detector()
predictor = face_recognition.api.pose_predictor_68_point

face_detection_ = mediapipe.solutions.face_detection
drawing = mediapipe.solutions.drawing_utils

def mediapipe_inference(frame):
    with face_detection_.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            # print('\tmediapipe inference failed')
            return None
        else:
            # print('\tmediapipe inference done')

            for i, detection in enumerate(results.detections):
                x1 = int(round(detection.location_data.relative_bounding_box.xmin*frame.shape[1]))
                y1 = int(round(detection.location_data.relative_bounding_box.ymin*frame.shape[0]))
                x2 = int(round((detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width)*frame.shape[1]))
                y2 = int(round((detection.location_data.relative_bounding_box.ymin+detection.location_data.relative_bounding_box.height)*frame.shape[0]))
                
                
                return [x1, x2, y1, y2]
        
def dlib_inference(frame):
    org_image = frame
    image = org_image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    lip_points = []
    
    if len(rects) != 0:
        
        for (i, rect) in enumerate(rects):
            if i > 0:
                break
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # print('\tdlib inference done')
            x1 = shape.min(axis=0)[0]
            x2 = shape.max(axis=0)[0]
            y1 = shape.min(axis=0)[1]
            y2 = shape.max(axis=0)[1]
            
        
        return [x1, x2, y1, y2]
    else:
        # print('\tdlib inference failed')
        return None

def get_rectsize(x1,y1,x2,y2):
    w = abs(x2 - x1) + 1
    h = abs(y2 - y1) + 1
    cx = x1 + w // 2
    cy = y1 + h // 2

    return w*h, [cx, cy]



    

