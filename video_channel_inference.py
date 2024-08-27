from face_inference import dlib_inference, mediapipe_inference, get_rectsize

import cv2
import numpy as np
import random
from glob import glob
import os
import argparse
import json

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from speaking_detection_module import swin_binary
import time

import argparse

def frame_speaking_detect (frame, model):       
    dlib_shape = dlib_inference(frame)
    cv2_shape = mediapipe_inference(frame)

    rect_size_dlib, center_point_dlib = 0, None
    rect_size_cv2, center_point_cv2 = 0, None
    if dlib_shape is not None or cv2_shape is not None:
        if dlib_shape is not None:
            rect_size_dlib, center_point_dlib = get_rectsize(
                dlib_shape[0], dlib_shape[2], dlib_shape[1], dlib_shape[3]
            )
    
        if cv2_shape is not None:
            rect_size_cv2, center_point_cv2 = get_rectsize(
                cv2_shape[0], cv2_shape[2], cv2_shape[1], cv2_shape[3]
            )
    else:
        print("\tNo face detected")
        return None
    
    select_shape = dlib_shape if rect_size_dlib > rect_size_cv2 else cv2_shape
    select_center = center_point_dlib if rect_size_dlib > rect_size_cv2 else center_point_cv2
    
    face = frame[select_shape[2]:select_shape[3], select_shape[0]:select_shape[1]]

    try:
        face = cv2.resize(face, (224, 224))
        face = transform(face).cuda()

        probs = model(face.unsqueeze(0))
        probs = probs.cpu().detach().numpy()
        return probs
    except:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_ch0_path', type=str, default=f'compile_materials/opentalk/camera1_synced.mp4',
                        help='video ch0 path')
    parser.add_argument('--video_ch1_path', type=str, default=f'compile_materials/opentalk/camera2_synced.mp4',
                        help='video ch1 path')
    parser.add_argument('--video_ch2_path', type=str, default=f'compile_materials/opentalk/camera3_synced.mp4',
                        help='video ch2 path')

    parser.add_argument('--start_time', type=int, default=None,
                        help='start time(sec)')
    parser.add_argument('--end_time', type=int, default=None,
                        help='end time(sec)')  

    parser.add_argument('--inference_result_dict_path', type=str, default=f'compiled_sample/sample.json',
                        help='inference output path')
    parser.add_argument('--model_path', type=str, default='speaking_detection_model_weight.pth',
                        help='model path')
    args = parser.parse_args()

    os.makedirs(args.inference_result_dict_path.split('/')[-2], exist_ok=True)

    seed_number = 999
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    video_ch0 = cv2.VideoCapture(args.video_ch0_path) # wide channel
    video_ch1 = cv2.VideoCapture(args.video_ch1_path) # 1st channel
    video_ch2 = cv2.VideoCapture(args.video_ch2_path) # 2nd channel

    result_dict = {}

    # ================== Load model ==================
    model = swin_binary()
    
    checkpoint = args.model_path
    model = nn.DataParallel(model).cuda()
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.module.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((224, 224),),
    transforms.ToTensor()
    ])
    
    min_total_frame = min(video_ch1.get(cv2.CAP_PROP_FRAME_COUNT), 
                          video_ch2.get(cv2.CAP_PROP_FRAME_COUNT), 
                          video_ch0.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(args.start_time*video_ch0.get(cv2.CAP_PROP_FPS))+1 if args.start_time is not None else 0
    end_frame = int(args.end_time*video_ch0.get(cv2.CAP_PROP_FPS)) if args.end_time is not None else min_total_frame

    previous_channel = 0
    
    while (video_ch1.isOpened and video_ch2.isOpened and video_ch0.isOpened):
        ret_ch0, frame_ch0 = video_ch0.read()
        ret_ch1, frame_ch1 = video_ch1.read()
        ret_ch2, frame_ch2 = video_ch2.read()
        
        frame_n = int(video_ch1.get(cv2.CAP_PROP_POS_FRAMES))

        
        if frame_n < start_frame:
            continue
        elif frame_n > end_frame:
            break

        print("#Frame: {}/{} -------------------------".format(frame_n, int(min_total_frame))) if end_frame is None else print("#Frame: {}/{} -------------------------".format(frame_n, end_frame))

        frame_ch1 = cv2.resize(frame_ch1, (1280, 720))
        frame_ch2 = cv2.resize(frame_ch2, (1280, 720))
        

        probs_ch1=frame_speaking_detect(frame_ch1, model)
        probs_ch1 = probs_ch1.reshape(-1).tolist() if probs_ch1 is not None else None
        
        probs_ch2=frame_speaking_detect(frame_ch2, model)
        probs_ch2 = probs_ch2.reshape(-1).tolist() if probs_ch2 is not None else None

        # ================== probs compare ==================
        if probs_ch1 is None or probs_ch2 is None:
            selected_channel = previous_channel
        elif probs_ch1 is not None and probs_ch2 is not None:
            selected_channel = 1 if probs_ch1[1] > probs_ch2[1] else 2
            selected_channel = 0 if abs(probs_ch1[1] - probs_ch2[1]) < 0.05 else selected_channel
        elif probs_ch1 is None or probs_ch2 is None:
            selected_channel = previous_channel
        
        # ================== Write result ==================
        frame_result = {
            'frame_n': frame_n,
            'previous_channel': previous_channel,
            'selected_channel': selected_channel,
            'probs_ch1': probs_ch1, 
            'probs_ch2': probs_ch2 
        }
        result_dict[frame_n] = frame_result

        
        with open(args.inference_result_dict_path, 'w') as f:
            json.dump(result_dict, f ,indent=4)

        previous_channel = selected_channel
        print("\tprevious_channel: ", previous_channel)
        print("\tselected_channel: ", selected_channel)
        print("\tprobs_ch1: ",probs_ch1)
        print("\tprobs_ch2: ",probs_ch2)
        
        if frame_n >= min_total_frame:
            break
        







