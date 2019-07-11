#!/usr/bin/env python
# coding: utf-8

# In[55]:


from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import scipy.linalg as la


# In[63]:


#### untils
# Moving Average filter
def MA(seq, w):
    """
    Moving average filter
        seq: sequence;
        w: window size, should be odd
    """
    out0 = np.convolve(seq,np.ones(w,dtype=int),'valid')/w
    r = np.arange(1,w-1,2)
    start = np.cumsum(seq[:w-1])[::2]/r
    stop = (np.cumsum(seq[:-w:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

#### calculate bias
def TH(x, th=0.1):
    """
    Thresholding function: erase insignificant noises by thresholding
        th is the relative threshold -- ratio to the largest abs value
    returned values are between 0 and 1 after max abs normalization
    """

    x_n = x/np.abs(x).max()
    res = x_n.copy()
    res[np.abs(res)<th] = 0
    return res

def shrink(x, scale=0.01, mode='reciprocal'):
    """
    Shrink function
        shrink larger values more and smaller values less
        mode:
            'reciprocal': sign(x)*|x|/(|x|+1)
            'sqrt': sign(x)*(sqrt(|x|))
            'logistic': exp(x)/(exp(x)+1) - 1/2
            'tanh': tanh(x)
            o.w.: scale only, no shrinkage
    """

    if mode == 'reciprocal':
        res = np.sign(x)*np.abs(x)/(np.abs(x)+1)
    elif mode == 'sqrt':
        res = np.sign(x)*np.sqrt(np.asb(x))
    elif mode == 'logistic':
        res = np.exp(x)/(np.exp(x)+1) - 1/2
    elif mode == 'tanh':
        res = np.tanh(x)
    else:
	res = x
    res *= scale

    return res


def est_bias(acc, ma_size=101, th=0.1, scale=0.01):
    """
    Calculate bias of observed pitch based on the information of acceleration:
        following the pipeline as
            acceleration --> MA --> TH --> shrink --> bias of pitch
    """
    
    bias = MA(acc, ma_size)
    bias = TH(bias, th)
    bias = shrink(bias, scale)
    return bias


# In[87]:


class pitch_model(object):
    def __init__(self, df):
        # parameters
        self.df = df
        self.x = df.utm_east
        self.y = df.utm_north
        self.z = df.utm_up
        self.roll = df.roll
        self.pitch = df.pitch
        self.yaw = df.yaw
        self.v = df.velo_robot_x
        self.acc = df.acc_robot_x
        self.sigma = 300 # used for gaussian filter
        
    def pre_pitch(self, shift=None):
        """
        Preprocess pitch:
            convert to [-pi, pi] and erase location shift of observation
        """

        pitch = self.pitch
        pitch_r = pitch.copy()
        pitch_r[pitch>np.pi] = 2*np.pi - pitch[pitch>np.pi]
        pitch_r[pitch<=np.pi] = - pitch[pitch<=np.pi]

        # if no bias is specified, we resort to the assumption of 0 pitch expectation
        if shift == None:
            shift = np.mean(pitch_r)

        pitch_r = pitch_r - shift
        self.pitch = pitch_r
        
    def fit(self, sigma = None):
        if sigma == None:
            sigma = self.sigma
        bias = est_bias(self.acc)
        pitch_rb = self.pitch - bias # pitch after removing estimated bias
        pitch_smooth = gaussian_filter1d(pitch_rb, sigma, mode='nearest')
        self.pitch_smooth = pitch_smooth
        self.pitch_bias = bias
        self.pitch_reducedbias = pitch_rb
        self.pitch_gaussiannoise = pitch_rb - pitch_smooth

