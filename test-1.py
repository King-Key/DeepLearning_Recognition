#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-17 09:47:21
# @Author  : WangGuo
# @Email   : guo_wang_113@163.com
# @Github  : https://github.com/King-Key

#已有模型的使用

import numpy as np 
import os 
import requests
import sys
import tarfile
import tensorflow as tf 
import zipfile
import urllib


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt 
from PIL import Image

#使用的模型名称，下载
MODEL_NAEM='ssd_mobilenet_vl_coco_11_06_2017'
MODEL_FILE=MODEL_NAEM+'.tar.gz'
DOWNLOAD_BASE='https//download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT=MODEL_NAEM+'/frozen_inference_graph.pb'

PATH_TO_LABELS=os.path.join('data','mscoco_label_map.pbtxt')
NUM_CLASSES=90

opener=urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE+MODEL_FILE,MODEL_FILE)
tar_file=tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
	file_name=os.path.basename(file.name)
	if 'frozen_inference_graph' in file_name:
		tar_file.extract(file,os.getcwd())

#新建一个图
detection_graph=tf.Graph()
with detection_graph.as_default():
	od_graph_def=tf.GraphDef()

	with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
		serialized_graph=fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name='')

#将index（数字）转换为类别名
label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
categories_index=label_map_util.create_category_index(categories)
#将图片转为numpy数组的形式
def load_image_imto_numpy_array(image):
	(im_width,im_height)=image.size 
	return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

#定义检测的图片
PATH_TO_TEST_IMSGES_DIR='test_images'
TEST_IMAGE_PATH=[os.path.join(PATH_TO_TEST_IMSGES_DIR,'image{}.jpg'.format(i)) for i in range(1,3)]
#输出图像的大小
IMAGE_SIZE=(12,8)

#检测
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		for image_path in TEST_IMAGE_PATHS:
			image=Image.open(image_path)

            #图片格式转换
			image_np=load_image_imto_numpy_array(image)

            #图片扩展一维
			image_np_expandes=np.expand_dims(image_np,axis=0)
			image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')

			#存放所有检测狂
			boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            
            #表示结果的confidence
			scores=detection_graph.get_tensor_by_name('detection_scores:0')

            #表示框所对应的类别
			classes=detection_graph.get_tensor_by_name('detection_classes:0')

            #表示检测框的个数
			num_detection=detection_graph.get_tensor_by_name('num_detections:0')

			#开始计算
			(boxes,scores,classes,num_detections)=sess.run([boxes,scores,classes,num_detections],feed_dict={image_tensor:image_np_expandes})

            #可视化
			vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_noemalized_coordinaties=True,line_thickness=8)
			plt.figure(figsize=IMAGE_SIZE)
			plt.imshow(image_np)

