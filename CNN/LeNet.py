# -*- coding: utf-8 -*-
# @Time    : 2019/12/3 10:16
# @Author  : King-Key
# @Email   : guo_wang_113@163.com
# @Blogs   : kingkey.club


import tensorflow as tf
import tensorflow.keras as keras

#使用keras.Sequential创建LeNet-5
network=keras.Sequential([
    keras.layers.Conv2D(6,kernel_size=3,strides=1), #第一层卷积，6*（3*3）*1,6个卷积核，卷积核大小为3*3，步长为1
    keras.layers.MaxPooling2D(pool_size=2,strides=2), #池化层，
    keras.ReLU(),#激活函数
    keras.layers.Conv2D(16,kernel_size=3,strides=1), #第二层卷积
    keras.layers.MaxPooling2D(pool_size=3,strides =2), #第二层池化
    keras.layers.ReLU(), #激活函数
    keras.layers.Flatten(), #打平层，方便全连接层处理
    keras.layers.Dense(120,activation ="relu")
])
