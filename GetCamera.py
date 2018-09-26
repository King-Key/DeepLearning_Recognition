#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2018-09-12 02:03:27
# @Author  : WangGuo (guo_wang_113@163.com)
# @Link    : ${link}
# @Version : $Id$

#Get camera shooting authority
import cv2
import time

def GetCamera():
	cap=cv2.VideoCapture(0)


def ShowCamera(cap):
	while (1):
		ret ,frame=cap.
		cv2.imshow("Cap", frame)
		pass

if __name__=='__main__':
	cap=GetCamera()
	ShowCamera(cap)


