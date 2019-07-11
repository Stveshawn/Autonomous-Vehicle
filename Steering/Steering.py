#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import collections as c
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la
import time
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse


# utils
rmse = lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)) # n is almost the same as (n-1)
mae = lambda y, y_pred: np.mean(abs(y - y_pred)) # n is almost the same as (n-1)


def plot_cov_ellipse(cov, pos, q=None, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : (2, 2) array
            Covariance matrix.
        q : float, optional
            Confidence level, should be in (0, 1)
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    
    if q is not None:
        q = np.asarray(q)
    elif nstd is not None:
        q = 2 * norm.cdf(nstd) - 1
    else:
        raise ValueError('One of `q` and `nstd` should be specified.')
    r2 = chi2.ppf(q, 2)
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1,0]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals * r2) # sqrt(lambda_i) * r
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip



class Steering(object):
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
        self.v_diff = self.v.diff() * 10
        self.steering_status = data_new.steering_status
        self.omega_yaw = data_new.omega_yaw
        self.L = 0.73    
        self.str_time = np.array(list(map(lambda x:time.strftime('%Y/%m/%d', time.localtime(x)), self.time)))
    
    
    def filter_obs(self, v_th=0.7, cmd_th=0, steering_th=300, drop_bad_bags=True, bag_auto_th=0.5):
        """
        Filter out bad steering observations to have valid observations where
            cmd > cmd_th
            v > v_th
            acc != 0
            abs(steering_status) < steering_th
            system_status == 1
            drop bags which have manually operations of more than 50%
            
        drop_bad_bags: where we want to drop bags which have too many manual operaitons
        """
        
        # basic filtering, delete bad/incorrect observations
        idx = self.cmd > cmd_th
        idx &= self.v > v_th
        idx &= self.acc != 0
        idx &= self.system_status == 1
        idx &= abs(self.steering_status) < steering_th
        print('Progressing ..', end='\r')
        
        # optional filtering, delete trips with too many manual operation
        status_ratio = lambda x: np.mean(self.system_status[self.bag_name == x] & 1)
        bns = list(c.Counter(self.bag_name).keys())
        # out_bags = [(bn,status_ratio(bn), sum(bag_name == bn)) for bn in bns if status_ratio(bn)<1]
        out_bags = [bn for bn in bns if status_ratio(bn)<0.5]
        print('Progressing ....', end='\r')
        for i, out_bag in enumerate(out_bags):
            idx &= (self.bag_name != out_bag)
        self.idx = idx
        print('Progressin ......', end='\r')
        data2 = self.data[idx].copy()
        self.data2 = data2.reset_index(drop=True)
        self.update_vars(self.data2) # update all variables based on data2
        self.fwa = np.arctan(self.omega_yaw * self.L / self.v) # front wheel angle
        print('Done. {:.2f} % observations left after filtering ({} out of {}).'.format(sum(idx)/len(idx)*100, sum(idx), len(idx)))
        
    def fit_by_batches(self, by='equally_spaced', n_batch=100, min_bagsize=500, plot_paras=True):
        """
        Get smaller chronologically sorted data batches and
        regress (front wheele angle ~ steering_status) on each of them.
        Args:
            by: the way of seperating batches, 'equally_spaced' or 'by_bag',
                where 'equally_space' means seperate the data as batches of 
                equal size, 'by_bag' means seperate by each bag_name.
            n_batch: number of batches when by 'equally_spaced', ignored when by 'by_bag'
            min_bagsize: minimal size of bagsize required to be a valid batch, ignored
                when by 'equally_spaced'
        """
        
        self.by = by
        if by == 'equally_spaced':
            batch_size = self.n // n_batch + 1
            self.batch_size = batch_size
            X = np.c_[np.ones(self.n), self.steering_status]
            y = self.fwa
            
            Paras = []
            for i in range(n_batch):
                X_tmp = X[i*batch_size:(i+1)*batch_size]
                y_tmp = y[i*batch_size:(i+1)*batch_size]
                batch_name = 'batch {:d}: from {:s} to {:s}'.format(i,
                                                                    self.str_time[i*batch_size],
                                                                    self.str_time[i*batch_size+len(y_tmp)-1])
                para_tmp = la.inv(X_tmp.T @ X_tmp) @ X_tmp.T @ y_tmp
                sig_tmp = rmse(X_tmp @ para_tmp, y_tmp)
            #     para_se_tmp = np.sqrt(np.diag(la.inv(X_tmp.T @ X_tmp) * sig2_tmp))
            #     ss = para_tmp / para_se_tmp
                Paras.append((batch_name, *para_tmp, sig_tmp))
        
        elif by == 'by_bag':
            bn_count = dict(c.Counter(self.bag_name))
            from_bn_2_size = lambda x: bn_count[x]
            bag_size = np.array(list(map(from_bn_2_size, self.bag_name)))

            bag_name_tmp = self.bag_name[bag_size>500] # bag_name's whose size > 500
            fwa_tmp = self.fwa[bag_size>500]
            steering_status_tmp = self.steering_status[bag_size>500]
            bns = list(c.Counter(bag_name_tmp))
            
            X = np.c_[np.ones(len(steering_status_tmp)), steering_status_tmp]
            y = fwa_tmp
            
            Paras = []
            self.bns = bns
            for i, bn_ in enumerate(bns):
                batch_name = 'batch {:d}: {:s}'.format(i, bn_)
                idx_tmp = (bag_name_tmp == bn_)
                X_tmp = X[idx_tmp]
                y_tmp = y[idx_tmp]
                para_tmp = la.inv(X_tmp.T @ X_tmp) @ X_tmp.T @ y_tmp
                sig_tmp = rmse(X_tmp @ para_tmp, y_tmp)
            #     para_se_tmp = np.sqrt(np.diag(la.inv(X_tmp.T @ X_tmp) * sig2_tmp))
            #     ss = para_tmp / para_se_tmp
                Paras.append((batch_name, *para_tmp, sig_tmp))
        
        else:
            raise('argument "by" should be in ["equally_spaced", "by_bag"]!')
            
        df_paras = pd.DataFrame(Paras, columns = ['batch', 'Int_', 'Coef_', 'Sig_'])
        
        if plot_paras:
            # steering_status: left +; right -
            plt.figure(figsize=(15, 9))
            plt.subplot(311)
            plt.plot(df_paras.Int_, marker = '.')
            plt.title('Estimated parameters when fitting by %s across time.' % by)
            plt.axvline(36)
            plt.ylabel('Int.')
            plt.subplot(312)
            plt.plot(df_paras.Coef_, marker = '.')
            plt.axvline(36)
            plt.ylabel('Coef.')
            plt.subplot(313)
            plt.plot(df_paras.Sig_, marker = '.')
            plt.ylabel(r'$\hat\sigma^2$(MSE)')
            plt.xlabel('Index for time intervals');
            
        self.df_paras = df_paras
        
        

    def plot_batch(self, i):
        """
        plot the scatter plot and regression line for the i-th batch
        """
        
        if self.by == 'equally_spaced':
            x_tmp = self.steering_status[i*self.batch_size: (i+1)*self.batch_size]
            y_tmp = self.fwa[i*self.batch_size: (i+1)*self.batch_size]
        elif self.by == 'by_bag':
            x_tmp = self.steering_status[self.bag_name == self.bns[i]]
            y_tmp = self.fwa[self.bag_name == self.bns[i]]
        
        plt.plot(x_tmp, y_tmp, '.', label = self.df_paras.batch[i], alpha=0.3)
        plt.plot(x_tmp, x_tmp*self.df_paras.Coef_[i]+self.df_paras.Int_[i], label = self.df_paras.batch[i])
        plt.xlabel('steering_status')
        plt.ylabel('front wheel angle')
        
    def plot_batches(self, *args, fsize = (15,10)):
        """
        Plot several batches for comparison
        """
        
        plt.figure(figsize=fsize)
        for i in args:
            self.plot_batch(i)
        plt.legend()
        
        
# Some parts of the analysis in not included in this file since they are not relatively trivial:
# refer to the jupyter-notebook for more EDAs and exploration on distributions
        
###################################################################################
################ Bayesian Model for Outlier Detection #############################
###################################################################################


    def BLR(self, X, y, phi0=50000, a0=1/2, b0=1/100000, plot_region=True, alpha=0.05):
        """
        Build a Bayesian Linear Regression Model with conjugate priors (Unit Information 
        under normal-gamma scheme) to detect outliers with credible intervals/regions
        
        If plot_region: plot the credible region given a significance level, alpha
                  else: plot the marginal credible intervals
        """
        
        # Unit information prior
        n = len(X)
        Lam0 = n*phi0*la.inv(X.T @ X)
        beta_ols = la.inv(X.T @ X) @ X.T @ y
        mu0 = beta_ols
        self.BLR_priors = (a0, b0, mu0, Lam0)
        
        # Posterior parameters
        mu_n = la.inv(X.T @ X + Lam0) @ (Lam0 @ mu0 + X.T @ X @ beta_ols)
        Lam_n = X.T @ X + Lam0
        a_n = a0+n/2
        b_n = b0+1/2*(y.T @ y + mu0.T @ Lam0 @ mu0 - mu_n.T @ Lam_n @ mu_n)
        self.BLR_posteriors = (a_n, b_n, mu_n, Lam_n)
        
        if plot_region:
            plt.figure(figsize=(15,10))
            xx, yy = self.df_paras[['Int_', 'Coef_']].values.T
            plt.plot(xx, yy, '.')
            plt.plot(*mu_n, 'ro', label = 'pos. mean')
            # Plot a transparent 2 standard deviation covariance ellipse
            ellip = plot_cov_ellipse(cov=la.inv(Lam_n), pos=mu_n, q = 1-alpha, color='green', alpha=0.3)
#             self.ellip = ellip
            plt.legend();
            cos_angle = np.cos(np.radians(180.-ellip.angle))
            sin_angle = np.sin(np.radians(180.-ellip.angle))

            xc = xx - ellip.center[0]
            yc = yy - ellip.center[1]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle 

            rad_cc = (xct**2/(ellip.width/2.)**2) + (yct**2/(ellip.height/2.)**2)
            out_idx = rad_cc>1
            self.outliers = self.df_paras[out_idx]
            
        else:
            upr = mu_n + norm.ppf(0.975) * np.sqrt(np.diag(la.inv(Lam_n)))
            lwr = mu_n - norm.ppf(0.975) * np.sqrt(np.diag(la.inv(Lam_n)))
            plt.figure(figsize=(15,10))
            xx, yy = self.df_paras[['Int_', 'Coef_']].values.T
            plt.plot(xx, yy, '.')
            plt.plot(*mu_n, 'ro', label = 'pos. mean')
            plt.axvline(upr[0], linestyle = '--', color='g', label = 'credible interval')
            plt.axvline(lwr[0], linestyle = '--', color='g')
            plt.axhline(upr[1], linestyle = '--', color='g')
            plt.axhline(lwr[1], linestyle = '--', color='g')
            plt.legend();
            
            out_idx = ((xx>upr[0]) | (xx<lwr[0]) | (yy>upr[1]) | (yy<lwr[1]))
            self.outliers = self.df_paras[out_idx]

