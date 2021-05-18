############################################################################
# TEST one sinle image
# This script uses one sinle image as input, generate bounding box with id
# To test on video, please use main.py
############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import shutil
import argparse
import numpy as np

sys.path.append("../../../common/")

from dataloader import LoadVideo, LoadImages
from multitracker import JDETracker
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis

from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource 

import math
import operator
import functools 
from PIL import Image

"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-12-11 10:12:13
MODIFIED: 2020-12-11 14:04:45
"""
def image_contrast(image1, image2):
    """
    Verify that the pictures are the same
    """
    file1 = Image.open(image1)
    file2 = Image.open(image2)
    h1 = file1.histogram()
    h2 = file2.histogram()
    ret = math.sqrt(functools.reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return ret

def test(opt):
    # Step 1: initialize ACL and ACL runtime 
    acl_resource = AclResource()

    # 1.2: one line of code, call the 'init' function of the AclResource object, to initilize ACL and ACL runtime 
    acl_resource.init()

    # Step 2: Load models 
    mot_model = Model('../model/mot_v2.om')

    dataloader = LoadImages(opt.test_img)

    # initialize tracker
    tracker = JDETracker(opt, mot_model, frame_rate=30)
    timer = Timer()
    results = []
    
    # img:  h w c; 608 1088 3
    # img0: c h w; 3 608 1088
    for frame_id, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0 and frame_id != 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking, start tracking timer 
        timer.tic()

        # list of Tracklet; see multitracker.STrack
        online_targets = tracker.update(np.array([img]), img0)

        # prepare for drawing, get all bbox and id
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()

        # draw bbox and id
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                        fps=1. / timer.average_time)
        cv2.imwrite(os.path.join('../data', 'test_output.jpg'), online_im)

    # verify if result is expected
    result = image_contrast('../data/test_output.jpg', opt.verify_img)
    print(result)
    if (result > 420 or result < 0):
        print("Similarity Test Fail!")
        sys.exit(1)
    else:
        print("Similarity Test Pass!")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_thres', type=float, default=0.35, help='confidence thresh for tracking')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--K', type=int, default=100, help='Max number of detection per image')

    parser.add_argument('--test_img', type=str, default='../data/test.jpg', help='path to the test image')
    parser.add_argument('--verify_img', type=str, default='../data/verify.jpg', help='path to the expected output image')

    opt = parser.parse_args()
    opt.mean = [0.408, 0.447, 0.470]
    opt.std = [0.289, 0.274, 0.278]
    opt.down_ratio = 4
    opt.num_classes = 1

    test(opt) 