#!/usr/bin/env python
# coding: utf-8



from data_pre import *
from Brake import *
import numpy as np


# load data
data = load_data()

# define class and process
m = Brake(data)
m.extract_brake_idx()
m.get_df(print_=False)
m.outlier_detect(one_sided=True)
m.outliers

for out_ in m.outliers.index.values:
    m.plot_detail(out_)

