#!/usr/bin/env python
# coding: utf-8


from scipy.stats import mode
import time
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
def to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s



class Brake(object):
    def __init__(self, data):
        self.n = len(data)
        self.bag_name = data.bag_name
        self.acc = data.acc
        self.v = data.v
        self.cmd = data.cmd
#         self.x = data.x
#         self.y = data.y
#         self.z = data.z
        self.pitch = np.sin(data.pitch)
        self.pitch2 = self.pitch/abs(self.pitch).max()
        self.system_status = data.system_status
        self.time = data.time # timestamp
        self.t = np.array(list(map(lambda x:time.strftime('%Y%m%d%H%M%S', time.localtime(x)), self.time)))
        self.dayofyear = np.array(list(map(lambda x:to_num(time.strftime('%j', time.localtime(x))), self.time)))
        self.month = np.array(list(map(lambda x:to_num(time.strftime('%m', time.localtime(x))), self.time)))
        self.week = np.array(list(map(lambda x:to_num(time.strftime('%W', time.localtime(x))), self.time)))

    def extract_brake_idx(self, eps_still=0.03, eps_move=0.3):
        """
        Identify and extract indices of braking samples in a given data frame
        By logic:

            start with `cmd==-57` and `v>eps_move`

            end if `v<eps_still`

            filter out invalid samples: 1. `acc==0`; 2. not in the same bag; 3. in manual driving mode;
        Returns: indices of brake samples in the original data frame.
        """

        # start: starting idx of brake behavior
        # end: ending idx of brake behavior
        start, end = 0, 0 # starting idx and ending idx
        eps_still = 0.03 # threshold of being still
        eps_move = 0.3 # threshold of driving (initial state)
        ind = 0 # indicator of whether in the process of brake behavior when iterating
        brake_idx = [] # tuples of indices of braking behavior
        for i in range(self.n):
            if (ind == 0) and (self.cmd[i] == -57) and (self.v[i] > eps_move) and (self.acc[i] != 0):
        #         cmd_tmp = self.cmd[i] # previous cmd to chech whether cmd increases
                ind = 1
                start = i
            elif ind == 1:
                if self.cmd[i] > -57:
                    ind = 0

            if (self.v[i] < eps_still) and (ind == 1):
                end = i
                ind = 0
                # status is odd, automated
                if self.bag_name[start] == self.bag_name[end] and (mode(self.system_status[start:end+1])[0] & 1 == 1):
                    # 2 ways of controlling braking
                    if self.cmd[start-1]>0:
                        brake_idx.append((start, end))
                    elif (self.cmd[start-3: start] == np.array([0, -20, -40])).all():
                        brake_idx.append((start-3, end))
            if i % 10000 == 0:
                print('Progressing: {:.2f} %'.format(i/self.n*100), end='\r')
        print('Done. %d brake samples extracted.' % len(brake_idx))
#         return brake_idx
        self.brake_idx = brake_idx
    
    
#################################### utils ####################################

# utils

    def plot_detail(self, seq, prior_ = 5, posterior_ = 1, t1 = 2, t2 = 3):
        """
        plot the detailed acc, v and cmd given a index in the brake_idx sequence
        args:
            seq: target brake_idx sequence number
            prior_: # of frames plotted before braking
            posterior_: # of frames plotted after braking
            t1: estimated time delay
            t2: time taken to increase to maximum abs acc.
        """

        idx = np.arange(self.brake_idx[seq][0]-prior_, self.brake_idx[seq][1]+1+posterior_)
        plt.figure(figsize=(10,6))
        plt.plot(self.v[idx], label='velo')
        plt.plot(self.cmd[idx]/57, marker='o', label='cmd rescaled')
        plt.plot(self.acc[idx], marker='o', label = 'acc')
        plt.plot(self.pitch2[idx], label ='pitch rescaled')
        plt.axhline(0)
        plt.axvline(self.brake_idx[seq][0], linestyle='--')
        plt.axvline(self.brake_idx[seq][0]+t1, linestyle='--')
        plt.axvline(self.brake_idx[seq][0]+t1+t2, linestyle='--')
        plt.xlabel('index in original data')
        plt.title('brake sample: # %d' % seq)
        plt.legend()

    def approx_brake_dis(self, start, end):
        """
        approximate the brake distance given a starting and ending index
        """

        vs = self.v[start: end+1]
        return sum(np.array([np.mean(vm) for vm in zip(vs, vs[1:])])*0.1)

    def acc_info(self, start, end, exclude = 5):
        """
        return the acceleration information (mean and min) given a starting and ending index
        """

        if end - start <= 6 and self.acc[end+1]>0:
            accs = self.acc[start+2+1 : end]
        else:
            accs = self.acc[start+exclude: end+1]
        return accs.mean(), accs.min()
    
###############################################################################


    def get_df(self, print_ = True):
        """
        get a tidy dataframe given a brake index sequence
        """
        brakes = []
        for i, idx in enumerate(self.brake_idx):
            brakes.append([self.v[idx[0]], idx[1]-idx[0], 2, 3, (idx[1]-idx[0]-5), 
                           self.approx_brake_dis(*idx), self.approx_brake_dis(idx[0], idx[0]+2),
                           self.approx_brake_dis(idx[0]+2, idx[0]+5), self.approx_brake_dis(idx[0]+5, idx[1]),
                           *self.acc_info(*idx), self.time[idx[0]], self.t[idx[0]],
                           self.dayofyear[idx[0]], self.month[idx[0]], self.week[idx[0]],
                           self.bag_name[idx[0]], np.mean(self.pitch[np.arange(*idx)])])
            if print_:
                print('seq:', i, '\nv:', self.v[idx[0]].round(4),
                      '\ttime:', (idx[1]-idx[0]),
                      '\tbrake_dist:', self.approx_brake_dis(*idx).round(4),
                      '\tacc: (mean: {:.4f}, max abs: {:.4f})'.format(*self.acc_info(*idx)),
                      '\n'+'-'*100)

        df_brakes = pd.DataFrame(brakes, columns = ['v', 't_total', 't1', 't2', 't3',
                                                    's_total', 's1', 's2', 's3', 'acc_mean',
                                                    'acc_min', 'timestamp', 'time', 'dayofyear',
                                                    'month', 'week', 'bag_name', 'pitch_mean'])
        df_brakes = df_brakes.sort_values(by = 'time')
        self.df_brakes = df_brakes
        
        
##############################################################################################
######### One-sided outlier detection based on t-test for `acc_mean` of braking.##############
##############################################################################################

    def outlier_detect(self, alpha=0.05, one_sided=True, plot=True):
        """
        Fit t-dist to 'acc_mean' of brake samples and apply one-sided test for outlier detection.
        """
        
        tdf = len(self.df_brakes) - 1
        tdist = stats.t(tdf)
        brake_acc = self.df_brakes.acc_mean # mean of acc in brake
        # brake_acc = self.df_brakes.acc_min # mean of acc in brake
        mu = brake_acc.mean()
        std = brake_acc.std()
        if one_sided:
            q = tdist.ppf(1-alpha)
            out_idx = (brake_acc - mu)/std > q
        else:
            q = tdist.ppf(1-alpha/2)
            out_idx = abs(brake_acc - mu)/std > q
        self.out_idx = out_idx
        self.outliers = self.df_brakes[out_idx].copy()
        
        if plot:
            plt.figure(figsize=(10,8))
            plt.subplot(211)
            plt.scatter(self.df_brakes.index, brake_acc, c=out_idx) # sorted by index
            plt.axhline(mu)
            plt.axhline(mu+q*std, linestyle = '--')
            if not one_sided:
                plt.axhline(mu-q*std, linestyle = '--')
            plt.ylabel("mean acceleration")
            plt.xlabel("index sorted by index")

            plt.subplot(212)
            plt.scatter(self.df_brakes.timestamp, brake_acc, c=out_idx) # sorted by timestamp
            plt.axhline(mu)
            plt.axhline(mu+q*std, linestyle = '--')
            if not one_sided:
                plt.axhline(mu-q*std, linestyle = '--')
            plt.ylabel("mean acceleration")
            plt.xlabel("timestamp")

