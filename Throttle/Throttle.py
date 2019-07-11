#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la
from scipy.stats import mode
from scipy import stats


# In[ ]:


# utils
def to_num(s):
    """
    convert to number if possible
    """
    
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

#
def is_int(x):
    """
    check if it is integer
    """
    
    try:
        return x-int(x) == 0
    except:
        return np.array(list(map(int, x)))-x==0
    
    
# other math functions
def KL(mu1, mu2, Sig1, Sig2):
    """
    KL divergence of two multivariate normal distribution
    """
    
    return (np.log(la.det(Sig2)/la.det(Sig1))-len(mu1)+np.matrix.trace(la.inv(Sig2)@Sig1)+(mu2-mu1).T@la.inv(Sig2)@(mu2-mu1))


def cond_dist_para(mu, Sig, cond):
    """
    get parameters for bivariate conditional normal distribution
    """
    
    res_mu = mu[1] + Sig[1,0]/Sig[0,0]*(cond - mu[0])
    res_sig = Sig[1,1] - Sig[1,0]/Sig[0,0]*Sig[0,1]
    return res_mu, res_sig


# In[ ]:


class Throttle(object):
    def __init__(self, data):
        self.data = data
        
    def update_vars(self, data_new):
        """
        Update corresponding values based on the new dataframe, data_new.
        """
                
        self.n = len(data_new)
        self.bag_name = data_new.bag_name
        self.acc = data_new.acc
        self.v = data_new.v
        self.cmd = data_new.cmd
        self.pitch = np.sin(data_new.pitch)
        self.pitch2 = self.pitch/abs(self.pitch).max()
        self.system_status = data_new.system_status
        self.time = data_new.time # timestamp
        if hasattr(data_new, 'v_diff'):
            self.v_diff = data_new.v_diff
        else:
            self.v_diff = self.v.diff() * 10
        self.steering_status = data_new.steering_status
        self.omega_yaw = data_new.omega_yaw
        
    def filter_obs(self, time_delay=3, th_acc=0.25, th_length=50, th_v_l=1.3, th_v_u=1.6):
        """
        Filter out "bad" throttle observations which we would not want in our model
        Args:
            time_delay: estimated time lay between cmd is sent and the actual accelerating happens
            th_acc: threshold for acceleration/v_diff, used for checking if the vehicle is in a steady state
            th_length: threshold for number of frames (100ms) needed to include a period of steady state in sample
            th_v_l: lower bound for vehicle's steady state
            th_v_u: upper bound for vehicle's steady state
        (default values are heuristic)
        Returns:
            A filtered data frame including valid throttle observations (moving in steady state).
        """
        
        tmp = []
        in_sample = False
        sample_grp = 0
        # cmd[i] ~ v[i+time_delay] ~ acc[i+time_delay]
        for i in range(self.n-time_delay):
            # should this frame be in sample?
            sample_ind = ((self.v[i+time_delay]>th_v_l) and (self.v[i+time_delay]<th_v_u) and 
                          (self.acc[i+time_delay]!=0) and (abs(self.v_diff[i+time_delay])<th_acc) and
                          (self.system_status[i] == 1) and (self.cmd[i]>20) and (self.cmd[i]<30) and (abs(self.steering_status[i])<300))
            if not in_sample:
                if sample_ind:
                    in_sample = True # whether is the state of steady motion
                    start_ = i + time_delay
            else:
                if sample_ind:
                    pass
                else:
                    end_ = i + time_delay # exclusive [start_, end_), index of v and acc ~ cmd[i+time_delay]
                    in_sample = False
                    if((end_ - start_) > th_length) and (self.bag_name[start_-time_delay] == self.bag_name[end_]):
                        tmp.append([sample_grp, (start_, end_)])
                        sample_grp += 1
            if i % 10000 == 0:
                print('Filtering out bad observations: {:.2f} %'.format(i/self.n*100), end='\r')
        print('Finished filtering out bad observations.            ', end='\n')
                
        df = []
        for i in range(len(tmp)):
            idx = np.arange(*tmp[i][1])
            sample_grp = tmp[i][0]
            for idx_ in idx:
                df.append([self.bag_name[idx_-time_delay], self.time[idx_-time_delay], self.v[idx_],
                           self.cmd[idx_-time_delay], self.acc[idx_], self.v_diff[idx_],
                           self.pitch[idx_], self.system_status[idx_-time_delay], self.omega_yaw[idx_],
                           self.steering_status[idx_-time_delay], sample_grp])
            print('Parsing new dataframe: {:.2f} %'.format(i/len(tmp)*100), end='\r')
        print('Finished parsing new dataframe.               ', end='\n')
        self.data2 = pd.DataFrame(df, columns = ['bag_name', 'time', 'v', 'cmd', 'acc', 'v_diff',
                                            'pitch' , 'system_status', 'omega_yaw', 'steering_status', 'sample_grp'])
        
        
############################################################################################
############################################# Fit by Batches ###############################
############################################################################################


    def fit_by_batches(self, n_batch=100, plot_EDA=True):
        """
        Split the data into chronologically sorted and equally spaced batches
        and estimate a bivariate normal distribution P(v, cmd|acc=0) to each of them.
        Args:
            Whether or not plot EDA for fitted results:
            plot of estimated 2-D mean vectors for 100 batches
            pairwise KL divergence heatplot
            traceplot of parameters across different batches
        Add attrs to the object:
            Estimated parameters for 2-D normal joint distribution of P(v, cmd|acc=0) (for all batches)
        """
        

        batch_size = self.n // n_batch + 1
        self.batch_size = batch_size
        grp = self.data2.index.values // n_batch
        
        paras = []
        for i in range(n_batch):
            d_sub = np.array(self.data2[int(i*batch_size): int((i+1)*batch_size)][['v', 'cmd']])
            mu = d_sub.mean(axis = 0) # MLE
            Sig = (d_sub - mu).T @ (d_sub - mu) /(len(d_sub)-1) # MLE
            para = (mu, Sig)
            paras.append(para)
        mus = np.array(list(map(lambda x: x[0], paras))) # 100 * 2, n_samples * n_d
        Sigs = np.array(list(map(lambda x: x[1], paras))) # 100 * 2 * 2, n_samples * n_d * n_d
        
        if plot_EDA:
            # plot EDA: scatterplot of mean vectors and KL divergence
            KL_D = np.ones((n_batch, n_batch))
            for i in range(KL_D.shape[0]):
                for j in range(KL_D.shape[1]):
                    KL_D[i][j] = KL(mus[i], mus[j], Sigs[i], Sigs[j])
            plt.figure(figsize=(15,5.5))
            plt.subplot(121)
            plt.plot(mus[:,1], mus[:,0], linestyle='--', alpha=0.2)
            plt.scatter(mus[:,1], mus[:,0], c=np.arange(0,100), cmap='seismic')
            plt.title(r'scatterplot of $\mu$')
            plt.xlabel('cmd')
            plt.ylabel('v')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(KL_D, cmap='seismic')
            plt.colorbar()
            plt.title('pairwise KL divergence')
            plt.show()
            
            # plot EDA: traceplot of parameters for the joint distribution P(v, cmd|acc=0)
            plt.figure(figsize=(15,10))
            plt.subplot(211)
            plt.plot(mus[:,0], label = r'$\mu_{v}$')
            plt.plot(mus[:,0] + np.sqrt(Sigs[:,0,0]), c='gray', linestyle='--', alpha=0.3, label = r'$\pm\sigma_{v}$')
            plt.plot(mus[:,0] - np.sqrt(Sigs[:,0,0]), c='gray', linestyle='--', alpha=0.3)
            plt.legend()
            plt.title(r'Joint distribution parameteres ($\mu_v, \mu_{cmd}, \sigma_{v}, \sigma_{cmd}$)')
            plt.subplot(212)
            plt.plot(mus[:,1], label = r'$\mu_{cmd}$')
            plt.plot(mus[:,1] + np.sqrt(Sigs[:,1,1]), c='gray', linestyle='--', alpha=0.3, label = r'$\pm\sigma_{cmd}$')
            plt.plot(mus[:,1] - np.sqrt(Sigs[:,1,1]), c='gray', linestyle='--', alpha=0.3)
            plt.legend()
            
        self.mus = mus
        self.Sigs = Sigs
        
    def conditional_dist(self, v_target = 1.45, plot=True):
        '''
        Get the conditional distribution based on parameters for joint distributions over batches.
        '''
        
        paras_cond = []
        for i in range(len(self.mus)):
            paras_cond.append(cond_dist_para(self.mus[i], self.Sigs[i], v_target))
        paras_cond = np.array(paras_cond)
        
        if plot:
            plt.figure(figsize=(15,5))
            plt.plot(paras_cond[:,0], marker='.')
            plt.title(r'Joint distribution parameteres ($\mu_{cmd}, \sigma_{cmd}|v=1.45$)')
            plt.plot(paras_cond[:,0] + paras_cond[:,1], c='gray', linestyle='--', alpha=0.3)
            plt.plot(paras_cond[:,0] - paras_cond[:,1], c='gray', linestyle='--', alpha=0.3)
            plt.axhline(25, linestyle='--', c='r', alpha=0.2)
            plt.axvline(36, linestyle='--', c='r', alpha=0.2)
        self.paras_cond = paras_cond
        
############################################################################################
########################################## Outlier Detection ###############################
############################################################################################


    def outlier_detect(self, data_base, data_new=None, v_target=1.45, alpha=0.05):
        """
        Detect outliers of `command` with CI of the conditional normal distribution
        P(cmd|v=v_target, acc=0) estimated from data_base.
        And give detection results for new data `data_new`.
        Args:
            data_base: N*2 d array, containing observations of 'v' and 'cmd'
            data_new: M*2 d arary, containing observations of 'v' and 'cmd'
                if data_new = None, only the CI is given
            v_target: target steady state velocity
            alpha: significance level
        """


        mu = data_base.mean(axis=0).values
        Sig = np.cov(data_base.values.T)

        def cond_CI(mu, Sig, v_target = 1.45, alpha=0.05):
            """
            give the 95% confidence interval of cmd given a expected v
            """
            mu_tmp, sig_tmp = cond_dist_para(mu, Sig, v_target)
            z = stats.norm.ppf(1-alpha/2)
            return (mu_tmp - z*sig_tmp, mu_tmp + z*sig_tmp)

        self.CI = cond_CI(mu, Sig, v_target)
        self.v_target = v_target
        print('The {}% confidence interval for cmd is [{}, {}] when v_target={}.'.format((1-alpha)*100, *self.CI, v_target))
        
        
        if data_new is not None:
            return (data_new.v.values < self.CI[0]) | (data_new.v.values > self.CI[1])

