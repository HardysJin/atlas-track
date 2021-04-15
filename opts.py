from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--conf_thres', type=float, default=0.35, help='confidence thresh for tracking')
    self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
    self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
    self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    
    self.parser.add_argument('--input-video', type=str, default='inputs/london_t.mp4', help='path to the input video')
    self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output-root', type=str, default='outputs', help='expected output root path')
    


  def init(self):
    opt = self.parser.parse_args()
    opt.K = 500
    opt.mean = [0.408, 0.447, 0.470]
    opt.std = [0.289, 0.274, 0.278]
    opt.down_ratio = 4
    opt.num_classes = 1
    return opt
