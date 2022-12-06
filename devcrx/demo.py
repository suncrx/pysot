# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:48:06 2022

@author: renxi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

#----------------------------------------------
CFG = '../experiments/siamrpn_r50_l234_dwxcorr/config.yaml'    
SNAPSHOT = '../experiments/siamrpn_r50_l234_dwxcorr/model.pth'

CFG = '../experiments/siamrpn_alex_dwxcorr/config.yaml'    
SNAPSHOT = '../experiments/siamrpn_alex_dwxcorr/model.pth'

CFG = '../experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml'    
SNAPSHOT = '../experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth'

#CFG = '../experiments/siammask_r50_l3/config.yaml'    
#SNAPSHOT = '../experiments/siammask_r50_l3/model.pth'


#---------------------------------------------
VIDEO = 'D:/GeoData/Benchmark/VIDEOS/VTB/Car24/img'
GT_FILE = 'D:/GeoData/Benchmark/VIDEOS/VTB/Car24/groundtruth.txt'

VIDEO = 'D:/GeoData/Benchmark/VIDEOS/VOT2018/road/img'
GT_FILE = None

#VIDEO = 'D:/GeoData/Videos/Drone/drone1/frames2'
#GT_FILE = 'D:/GeoData/Videos/Drone/drone1/groundth.txt'

#VIDEO = 'D:\GeoData\Videos\Drone\drone_traffic04.mp4'
#GT_FILE = None

#VIDEO = 'D:/GeoData/Videos/Drone/dji0265/DJI_0265.mp4'
#GT_FILE = None

#VIDEO = 'D:/GeoData/Videos/TestVideos/street-5025.mp4' #'People-6387.mp4'
#GT_FILE = None

#VIDEO = '../demo/bag.avi'
#GT_FILE = None
#-------------------------------------------------

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', 
                    default = CFG)
parser.add_argument('--snapshot', type=str, help='model name',
                    default = SNAPSHOT)
parser.add_argument('--video_name', type=str, help='videos or image files',
                    default = VIDEO)
parser.add_argument('--gt_file', type=str, help='groundtruth filepath',
                    default = GT_FILE)
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame 
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        #images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images= sorted(images)
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    #cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    
    frame_count = 0
    for frame in get_frames(args.video_name):
        frame_count += 1
        if frame_count==1:
            # select ROI 
            if args.gt_file is None or not os.path.exists(args.gt_file):
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
            # Read ROI from groundtruth file
            else:
                anno = np.loadtxt(os.path.join(args.gt_file), delimiter=',')
                init_rect = anno[0]
                
            tracker.init(frame, init_rect)
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                #cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                #              True, (0, 255, 0), 1)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.5, mask, 0.5, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 2)
            
            cv2.putText(frame, 'Frame %d' % frame_count, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0,0,255))               
            cv2.imshow(video_name, frame)
            k = cv2.waitKey(10)
            if k == 27:
                break
            
    cv2.destroyAllWindows()            

if __name__ == '__main__':
    main()
