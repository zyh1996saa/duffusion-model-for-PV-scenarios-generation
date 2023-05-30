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

print('*'*30+'data loading' + '*'*30)

days_series = np.load(r'.\days_series.npy')

day_length = days_series.shape[1]

start_point,end_point = 0,143

for i in range(day_length):   
    if days_series[:,i].max()>0:
        start_point = i
        break
    
for i in range(day_length):
    if days_series[:,day_length-i-1].max()>0:
        end_point = day_length-i-1
        break

data_shape = end_point - start_point

sample_size = days_series.shape[0]

# 采样标准正态分布
def sample_guass_noise(scale=1,size=(data_shape,)):
    return np.random.normal(loc=0,scale=1,size=size)

# 连乘alpha
def PI_alpha(alpha,t):
    temp = 1
    for i in range(t+1):
        temp *= alpha[i]
    return temp


# constance setting

repeat_time = 10 # 内存不够的话适当减少此项值，可变参数

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

t_for_each_data = np.random.randint(low=0, high=T, size=sample_size*repeat_time)



# forward diffusion process

ori_data =days_series[:,start_point:end_point]

for i in range(repeat_time-1):
    ori_data = np.vstack((ori_data,days_series[:,start_point:end_point]))
# diffusion_process
diff_tri = np.zeros((sample_size*repeat_time,data_shape))

noise_t = np.zeros((sample_size*repeat_time,data_shape))

print('*'*30+'diffusion process starts' + '*'*30)

for r in range(repeat_time*sample_size):

        t = t_for_each_data[r] #取出每一个数据的t值
        
        noise_t[r,:] = sample_guass_noise()
        
        diff_tri[r,:] = (x0_t_coeff[t]*ori_data[r]).reshape(1,data_shape) + (noise_t_coeff[t]*noise_t[r,:]).reshape(1,data_shape)
        
        print('\r%s/%s'%(r,repeat_time*sample_size),end='\r')

print('\n'*3)

print('*'*30+'dataset dividing' + '*'*30)

# 切割训练集
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(diff_tri,t_for_each_data, noise_t, test_size=0.1)

print('\n'*3)


print('*'*30+'defining neural networks' + '*'*30)
# 定义神经网络
"""
input0 = tf.keras.Input(shape=(data_shape,))
input1 = tf.keras.Input(shape=(1,))

x1 = tf.keras.layers.Dense(128,name='dense1',activation='gelu')(input1)
#x1 = tf.keras.layers.Dense(128,name='dense2',activation='gelu')(x1)
x1 = tf.keras.layers.Dense(96,name='dense3',)(x1)

x0 = tf.keras.layers.LSTM(
    256,
    activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    name='lstm'
)(tf.keras.layers.Reshape((1,data_shape))(input0))

x0 = tf.keras.layers.Dense(128,name='dense4',activation='gelu',)(x0)
x0 = tf.keras.layers.Dense(128,name='dense5',activation='gelu',)(x0)
x0 = tf.keras.layers.Dense(96,name='dense6',)(x0)
#x = tf.keras.layers.Concatenate(axis=1)([x0, x1])
x = tf.keras.layers.Attention(dropout=0.4)([x0, x1])
x = tf.keras.layers.Dense(256,name='dense7',activation='gelu',)(x)
x = tf.keras.layers.Dense(128,name='dense8',activation='gelu',)(x)
output = tf.keras.layers.Dense(data_shape,name='dense9',)(x)

# or use a simple network
"""
input0 = tf.keras.Input(shape=(data_shape,))
input1 = tf.keras.Input(shape=(1,))
x1 = tf.keras.layers.Dense(128,name='dense1',activation='gelu')(input1)
#x1 = tf.keras.layers.Dense(128,name='dense2',activation='gelu')(x1)
x1 = tf.keras.layers.Dense(96,name='dense3',)(x1)
x0 = tf.keras.layers.Dense(256,name='dense4',activation='gelu',)(input0)
x0 = tf.keras.layers.Dense(128,name='dense5',activation='gelu',)(x0)
x0 = tf.keras.layers.Dense(96,name='dense6',)(x0)
x = tf.keras.layers.Concatenate(axis=1)([x0, x1])
x = tf.keras.layers.Dense(256,name='dense7',activation='gelu',)(x)
x = tf.keras.layers.Dense(128,name='dense8',activation='gelu',)(x)
output = tf.keras.layers.Dense(data_shape,name='dense9',)(x)
model = tf.keras.Model([input0,input1], output)

model_input0 = X_train
model_input1 = T_train/T
model_output = y_train
model.compile(optimizer='Adam',loss='MSE')
model.fit([model_input0,model_input1],model_output,epochs=10000,batch_size=4096)


mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError(
    name='mean_absolute_error'
)

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0008,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Adam',
)

print('\n'*3)

model = tf.keras.Model([input0,input1], output)

model_input0 = X_train
model_input1 = T_train/T
model_output = y_train
model.compile(optimizer=opt,loss=mse,metrics=mae)





print('*'*30+'training' + '*'*30)

model.fit([model_input0,model_input1],model_output,epochs=int(4e4),batch_size=int(model_input0.shape[0]/128))

print('*'*30+'saving model' + '*'*30)

model.save('./model0224')
