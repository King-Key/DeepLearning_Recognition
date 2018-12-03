#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-17 09:31:14
# @Author  : WangGuo
# @Email   : guo_wang_113@163.com
# @Github  : https://github.com/King-Key

#关于opencv的一些使用

import cv2
#img=cv2.imread('../data/Aurora.jpg')

image=cv2.imread('/home/guo/图片/她-.jpg')
res=cv2.resize(image,(500,375),interpolation=cv2.INTER_CUBIC)
cv2.imshow('iker',res)
#cv2.imshow('image',image)
cv2.waitKey(0)



# #通过迭代器显示视频
# import os
# from itertools import cycle
# import cv2

# #列出文件下的所有图片
# filenames=os.listdir('frames')
# #生成无线循环的迭代器，每次迭代的对象都输出下一张图片
# img_iter=cycle([cv2.imread(os.sep.join(['frames',x])) for x in filenames])
# key=0
# while key!=27:
# 	cv2.imshow('视频',next(img_iter))
# 	key=cv2.waitKey(42)
# 	pass

