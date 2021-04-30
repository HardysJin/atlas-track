from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils import get_feat_from_idx
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided

# from https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy/54966908
def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def _nms(heat):
    # hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    hmax = pool2d(heat, 3, stride=1, padding=1)
    keep = (hmax == heat).astype(float)
    return heat * keep

def mot_decode(heat, wh, reg, ltrb=True, conf_thres=None):
    batch, cat, height, width = heat.shape

    # perform nms on heatmaps
    heat = _nms(heat.squeeze())
    
    # import matplotlib.pyplot as plt
    # plt.imshow(heat.squeeze())
    # plt.savefig("test.jpg", dpi=600)
    # raise Exception

    tmp = heat.flatten()

    topk_inds = np.nonzero(tmp > conf_thres)[0]

    scores = tmp[topk_inds]
    inds = topk_inds % (height * width)
    
    ys = (topk_inds / width) #.round()
    xs = (topk_inds % width).astype(float)

    clses = np.zeros(topk_inds.shape[0])
        
    reg = get_feat_from_idx(reg, inds)
    xs += reg[:,0]
    ys += reg[:,1]

    wh = get_feat_from_idx(wh, inds)

    detections = np.stack( [xs - wh[:, 0],
                            ys - wh[:, 1],
                            xs + wh[:, 2],
                            ys + wh[:, 3],
                            scores,
                            clses], axis=0)

    return detections.T, inds