# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:21:06 2023

@author: Yuhong Zhu

This is a special gift for my girlfriend Yixian Liu ~
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split


#导入原始数据集
os.chdir(r'.')

total_data = pd. read_csv('./total_data.csv')

# 构造序列集合

pv_name ='DHI'

day_length = 144

plant_num = 19

data_length = int(total_data.shape[0]/day_length) 

days_series = np.zeros((data_length,day_length))

# 出力有名值转换为标幺值
pv_max_value = np.array([
        total_data[total_data['label']==i][pv_name].max() for i in range(19) 
    ])


for day in range(data_length):
    
    day_series = total_data[pv_name][day*144:(day+1)*144]
    
    label = int(total_data['label'][day*144])
    
    days_series[day,:] = day_series/pv_max_value[label]
    
    print('\r%s/%s'%(day,data_length),end='\r')

# 保存处理后的数据集
np.save(r'.\days_series.npy', days_series)