from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./lib')

import os
import cv2
import shutil
import argparse

import numpy as np
from dataloader import LoadVideo, LoadImages
from multitracker import JDETracker
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis

#ACL model load and execute implementation
from acl_model import Model
#ACL init and resource management implementation
from acl_resource import AclResource 

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def demo(opt):
    # Step 1: initialize ACL and ACL runtime 
    acl_resource = AclResource()

    # 1.2: one line of code, call the 'init' function of the AclResource object, to initilize ACL and ACL runtime 
    acl_resource.init()

    # Step 2: Load models 
    mot_model = Model(acl_resource, 'model/dlav0.om')

    # Create output dir if not exist; default outputs
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    video_name = os.path.basename(opt.input_video).replace(' ', '_').split('.')[0]

    # setup dataloader, use LoadVideo or LoadImages
    dataloader = LoadVideo(opt.input_video, (1088, 608))
    # result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    # dir for output images; default: outputs/'VideoFileName'
    save_dir = os.path.join(result_root, video_name)    
    if save_dir and os.path.exists(save_dir):
        shutil.rmtree(save_dir) 
    mkdir_if_missing(save_dir)

    # initialize tracker
    tracker = JDETracker(opt, mot_model, frame_rate=frame_rate)
    timer = Timer()
    results = []
    
    # img:  h w c; 608 1088 3
    # img0: c h w; 3 608 1088
    for frame_id, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking, start tracking timer 
        timer.tic()

        # list of Tracklet; see multitracker.STrack
        online_targets = tracker.update(np.array([img]), img0)
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
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)


    # if opt.output_format == 'video':
    #     output_video_path = os.path.join(result_root, os.path.basename(opt.input_video).replace(' ', '_'))
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(frame_dir, output_video_path)
    #     os.system(cmd_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_thres', type=float, default=0.35, help='confidence thresh for tracking')
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--K', type=int, default=100, help='Max number of detection per image')

    parser.add_argument('--input_video', type=str, default='inputs/london_t.mp4', help='path to the input video')
    parser.add_argument('--output_format', type=str, default='video', help='video or text')
    parser.add_argument('--output_root', type=str, default='outputs', help='expected output root path')
    
    opt = parser.parse_args()
    opt.mean = [0.408, 0.447, 0.470]
    opt.std = [0.289, 0.274, 0.278]
    opt.down_ratio = 4
    opt.num_classes = 1

    demo(opt)
