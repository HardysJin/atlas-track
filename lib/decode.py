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

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    # hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    hmax = pool2d(heat, 3, stride=1, padding=1)
    keep = (hmax == heat).astype(float)
    return heat * keep
    
def _nmsv0(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def mot_decode(heat, wh, reg, ltrb=True, conf_thres=None):
    batch, cat, height, width = heat.shape

    # perform nms on heatmaps
    # print(heat.numpy.squeeze())
    heat = _nms(heat.squeeze())
    # heat = _nmsv0(heat)

    # import matplotlib.pyplot as plt
    # plt.imshow(heat.squeeze())
    # plt.savefig("test.jpg", dpi=600)
    # raise Exception

    # conf_thres = None

    tmp = heat.flatten()
    # tmp = heat.view(-1)

    # mask = tmp > conf_thres
    # print(mask.nonzero())
    # topk_inds = mask.nonzero()[...,-1]
    # topk_inds: tensor([17253, 17260, 17273, 17337, 17338, 17529, 17549, 17552, 17560, 17566, 17571, 17584, 17600, 17603, 17646, 17649, 17671, 17778, 17847, 17869, 17965, 18503, 18626, 19312, 19315])
    topk_inds = np.nonzero(tmp > conf_thres)[0]

    scores = tmp[topk_inds]
    # print(scores.shape)
    inds = topk_inds % (height * width)
    
    ys = (topk_inds / width) #.round()
    xs = (topk_inds % width).astype(float)
    # print(type(xs[0]))
    # print(topk_inds)
    # ys   = (topk_inds / width).int().float()
    # xs   = (topk_inds % width).int().float()
    # print(ys,xs)
    # tensor([63., 63., 63., 63., 63., 64., 64., 64., 64., 64., 64., 64., 64., 64., 64., 64., 64., 65., 65., 65., 66., 68., 68., 71., 71.]) 
    # tensor([117., 124., 137., 201., 202., 121., 141., 144., 152., 158., 163., 176., 192., 195., 238., 241., 263.,  98., 167., 189.,  13.,   7., 130.,   0.,   3.])
    clses = np.zeros(topk_inds.shape[0])
    # K = inds.size(0)
    # print(reg.shape, inds.shape)
        
    reg = get_feat_from_idx(reg, inds)
    # print("reg", reg.shape)
    xs += reg[:,0]
    ys += reg[:,1]

    wh = get_feat_from_idx(wh, inds)
    # wh 
    # if ltrb:
    #     wh = wh.view(batch, K, 4)
    # else:
    #     wh = wh.view(batch, K, 2)
    # clses = clses.view(batch, K, 1).float()
    # scores = scores.view(batch, K, 1)
    detections = np.stack( [xs - wh[:, 0],
                            ys - wh[:, 1],
                            xs + wh[:, 2],
                            ys + wh[:, 3],
                            scores,
                            clses], axis=0)

    return detections.T, inds


# tensor([[[115.64746,  58.96240, 119.86426,  68.47803,   0.43398,   0.00000],
#          [122.18872,  58.08496, 126.64380,  69.29980,   0.56937,   0.00000],
#          [135.09229,  57.56641, 139.76416,  69.71875,   0.36704,   0.00000],
#          [198.14355,  53.70703, 205.26270,  73.39453,   0.71303,   0.00000],
#          [198.73511,  53.77539, 205.86987,  73.29102,   0.71303,   0.00000],
#          [119.28467,  58.81409, 123.78076,  69.63831,   0.40592,   0.00000],
#          [139.04590,  58.04370, 143.90918,  70.64917,   0.35309,   0.00000],
#          [140.66846,  56.30078, 148.02979,  72.80859,   0.69884,   0.00000],
#          [149.17700,  56.34277, 155.75513,  72.83496,   0.74540,   0.00000],
#          [154.93750,  56.69824, 161.81055,  72.81543,   0.54432,   0.00000],
#          [160.10303,  56.23340, 166.61865,  73.17871,   0.46806,   0.00000],
#          [171.77979,  54.10156, 181.16260,  75.50000,   0.77662,   0.00000],
#          [188.93848,  54.52734, 196.09668,  74.30078,   0.36523,   0.00000],
#          [191.22607,  53.92480, 199.85889,  75.29199,   0.80776,   0.00000],
#          [233.43188,  52.92627, 243.34204,  76.23877,   0.75779,   0.00000],
#          [237.45874,  53.61304, 245.43921,  75.08960,   0.50684,   0.00000],
#          [257.31250,  50.13379, 269.73828,  79.15723,   0.52001,   0.00000],
#          [ 95.25928,  57.43359, 101.84131,  73.99609,   0.70089,   0.00000],
#          [163.72168,  56.48047, 171.42285,  74.58203,   0.56506,   0.00000],
#          [185.55176,  55.51270, 193.77051,  75.60645,   0.75274,   0.00000],
#          [ 10.72021,  60.45703,  16.01123,  72.47266,   0.40310,   0.00000],
#          [  4.68896,  59.72266,  10.73975,  77.41797,   0.57988,   0.00000],
#          [125.89502,  56.25708, 135.38330,  80.09302,   0.77696,   0.00000],
#          [ -8.63257,  56.82861,   3.07251,  86.14111,   0.47487,   0.00000],
#          [ -1.23682,  57.29663,   7.99756,  85.50757,   0.61509,   0.00000]]])