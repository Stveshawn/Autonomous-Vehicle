#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os
# utils
def to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s




def load_data(months = ['march', 'april', 'may']):
    """
    Load raw data by given months and make it a tidy data frame.
    requires data raw files from hadoop in the correct director
    """
    
    RES = []
    print('----start loading----')
    for month in months:
        file_parts = len(os.listdir(path='../data/'+month))-1
        for i in range(file_parts):
            fname = '../data/' + month + '/part-' + str(i).rjust(5, '0')
            print('{}: {:.2f}'.format(month, i/file_parts*100), end='%\r')
            with open(fname, 'r') as lines:
                count = 0
                ind_status, ind_pose = False, False
            #     numRows = 0
                for line in lines:
                    tmp = line.split()
            #         print(line.split())
                    if(tmp[1] == 'aw_idl/VehicleInput'):
                        res = []
                        res.extend(list(map(to_num, tmp)))
                        count += 1
            #             RES.append(res)
                    elif(tmp[1] == 'aw_idl/VehicleStatus'):
                        tmp_status = list(map(to_num, tmp))
                        ind_status = True
                    elif(count == 1):
                        tmp_pose = list(map(to_num, tmp))
                        ind_pose = True
                        count = 0
                    if ind_status and ind_pose:
                        res.extend(tmp_pose)
                        res.extend(tmp_status)
                        RES.append(res)
                        ind_status, ind_pose = False, False
        print(month, end =' done.               \n')
    print('----finish loading----') 
    
    
    cols = ['bag_name', 'topic', 'time', 'input_status', 'cmd', 'steering_input', 'bag_name2', 'topic2', 'time2', 'v', 'acc', 'x', 'y', 'z', 'pitch', 'omega_yaw', 'bag_name3', 'topic3', 'time3', 'system_status', 'speed_status', 'steering_status']
    data = pd.DataFrame(RES, columns = cols)
    data = data.drop(['topic', 'topic2', 'topic3', 'time2', 'time3', 'bag_name2', 'bag_name3'], axis=1)
    data = data.sort_values(by = 'time')
    data = data.reset_index(drop=True)
    return data

# data = load_data()

