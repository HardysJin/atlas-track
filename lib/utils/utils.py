import numpy as np

def get_feat_from_idx(feat, idx):
    ''' 
    feat: np array, NCHW
    idx:  np array of int
    '''
    feat = np.transpose(feat, (0, 2, 3, 1))
    feat = np.reshape(feat, (-1, feat.shape[-1]))
    feat = feat[idx]
    return feat