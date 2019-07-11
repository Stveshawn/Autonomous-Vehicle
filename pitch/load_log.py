#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import scipy as sp

# utils
def to_num(s):
    """
    convert str to num if possible:
        to int when appropriate, otherwise to float
    """
    
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
        
        
def log2df(file_name):
    """
    Get a dataframe from the log file of topic -- localized_pose
    """
    
    res = []
    with open(file_name, 'r') as lines:
        tmp = []
        for line in lines:
            if line.strip() != '---':
                if len((line.strip().split(':'))[1].strip()):
                    tmp.append(to_num((line.strip().split(':'))[1].strip()))
            else:
                res.append(tmp)
                tmp = []
                
    # columns specific for topic -- localized_pose
    cols = ['seq', 'secs', 'nsecs', 'frame_id', 'utm_east', 'utm_north',
           'utm_up', 'roll', 'pitch', 'yaw', 'velo_north',
           'velo_east', 'velo_down', 'velo_robot_x',
           'velo_robot_y', 'velo_robot_z', 'acc_robot_x',
           'acc_robot_y', 'acc_robot_z', 'omega_yaw', 'omega_pitch',
           'omega_roll', 'nav_mode', 'pos_mode', 'vel_mode']

    df = pd.DataFrame(res, columns=cols)
    df = df.drop(columns = ['seq', 'frame_id'])
    return df

