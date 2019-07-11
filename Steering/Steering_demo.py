#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from data_pre import *
from Steering import *


# In[2]:


data = load_data()


# In[3]:


m = Steering(data)
m.update_vars(m.data)
m.filter_obs()


# In[4]:


m.fit_by_batches(by='equally_spaced', plot_paras=False)


# In[5]:


m.plot_batches(36, 37)


# In[6]:


# use the most recent data as our samples for BLR
X = np.c_[np.ones(sum(m.str_time > '2019/04/16')), m.data2[m.str_time > '2019/04/16'].steering_status]
y = m.fwa[m.str_time > '2019/04/16']

m.BLR(X, y, plot_region=False)
m.outliers


# In[7]:


m.BLR(X, y)
m.outliers

