from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./lib')

# from utils import _init_paths
import logging
import os
import os.path as osp
from opts import opts
import shutil

from dataloader import LoadVideo
from track import eval_seq

#ACL model load and execute implementation
from acl_model import Model
#ACL init and resource management implementation
from acl_resource import AclResource 

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def demo(opt):
    # Step 1: initialize ACL and ACL runtime 
    acl_resource = AclResource()

    # 1.2: one line of code, call the 'init' function of the AclResource object, to initilize ACL and ACL runtime 
    acl_resource.init()

    # Step 2: Load models 
    mot_model = Model(acl_resource, 'model/dlav0.om')

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    new_name = os.path.basename(opt.input_video).replace(' ', '_').split('.')[0]

    dataloader = LoadVideo(opt.input_video, (1088, 608))
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, new_name)
    if frame_dir and osp.exists(frame_dir):
        shutil.rmtree(frame_dir) 
    eval_seq(opt, dataloader, mot_model, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=False)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, os.path.basename(opt.input_video).replace(' ', '_'))
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(frame_dir, output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
