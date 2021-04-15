import numpy as np
import os
import cv2
import sys
sys.path.append('./lib')
import argparse

#ACL model load and execute implementation
from acl_model import Model
#ACL init and resource management implementation
from acl_resource import AclResource 

import matplotlib.pyplot as plt

import dataloader
from tracking_utils.utils import mkdir_if_missing
# Path for MOT model
MODEL_PATH = 'model/dlav0.om'
# Video Path
# VIDEO_PATH = 'inputs/MOT17-11-SDP-raw.mp4'

def main(args):
    mkdir_if_missing(os.path.join(args.output_dir, os.path.basename(args.input_video)))
    
    # Step 1: initialize ACL and ACL runtime 
    acl_resource = AclResource()

    # 1.2: one line of code, call the 'init' function of the AclResource object, to initilize ACL and ACL runtime 
    acl_resource.init()

    # Step 2: Load models 
    mot_model = Model(acl_resource, MODEL_PATH)

    
    loader = dataloader.LoadVideo(args.input_video)

    # for i, img, img0 in loader:
    #     pass

    # Step 3: Face Detection Inference
    # load testing image file 
    image = cv2.imread('inputs/000058.png')
    # preprocessing the image file for face detection
    input_image = PreProcessing(image)
    
    img = np.moveaxis(input_image, 0, -1)
    plt.subplot(121)
    plt.imshow(img)

    # one line of code, use the 'Model' object for face detection, call its 'execute' function with parameter '[input_image]' and assign it to list 'resultList_face'
    ### Your code here, one line ###
    output = mot_model.execute([input_image])
    pred_hm = output[0][0]
    pred_hm = np.moveaxis(pred_hm, 0, -1)
    # print(np.min(pred_hm), np.max(pred_hm))
    plt.subplot(122)
    plt.imshow(pred_hm)
    plt.savefig('outputs/000058.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_video', type=str, default='inputs/MOT17-11-SDP-raw.mp4', help="test video path")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='expected output root path')
    parser.add_argument('--pb', type=str, default='../pretrained/dlav0_30', help="pb path")
    args = parser.parse_args()
    main(args)