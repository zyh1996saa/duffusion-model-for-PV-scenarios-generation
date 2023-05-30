# In[]
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

data_shape = 82

# 采样标准正态分布
def sample_guass_noise(scale=1,size=(data_shape,)):
    return np.random.normal(loc=0,scale=1,size=size)

# 连乘alpha
def PI_alpha(alpha,t):
    temp = 1
    for i in range(t+1):
        temp *= alpha[i]
    return temp

print('*'*30+'model loading' + '*'*30)
model = tf.keras.models.load_model('model0224')


T = 1000 # 可调参数
# 定义beta
beta_t = np.linspace(0.0001,0.02,T) #最好别调

# 定义alpha = 1- beta
alpha_t = 1 - beta_t

# 定义alpha_ba_t = alpha_1 * alpha_2 * ... * aplha_t 
alpha_ba_t = np.array([PI_alpha(alpha_t,t) for t in range(T)])

# 定义x0的系数 = sqrt(alpha_ba)
x0_t_coeff = np.sqrt(alpha_ba_t)

# 定义噪声z0的系数 = sqrt(1-alpha_ba)
noise_t_coeff = np.sqrt(1-alpha_ba_t)

#t_for_each_data = np.random.randint(low=0, high=T, size=sample_size*repeat_time)

sample_num = 1000

start_point = 32

end_point = 114

print('*'*30+'sampling' + '*'*30)

for i in range(sample_num):
    for t in range(T,0,-1):
        if t == T:
            xt = sample_guass_noise()
        else:
            xt = xt_red_1
        noise_t_pre = model.predict([xt.reshape(1,data_shape),np.array([[t/1000]])],verbose = 0).reshape(data_shape)
        alpha = alpha_t[t-1]
        alpha_ba = alpha_ba_t[t-1]
        z = sample_guass_noise()
        xt_red_1 = (1/np.sqrt(alpha))*(xt-(1-alpha)*noise_t_pre/np.sqrt(1-alpha_ba))+z*np.sqrt(1-alpha)
        print('\r t=%s,i=%s'%(t-1,i),end='\r')
    x = np.zeros((144,))
    x[start_point:end_point] = np.maximum(xt,0)
    np.save('./generated/data%s'%i,xt)