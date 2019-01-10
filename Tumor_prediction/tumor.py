#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-10 11:01:53
# @Author  : WangGuo
# @Email   : guo_wang_113@163.com
# @Github  : https://github.com/King-Key


#良/恶性乳腺癌肿瘤预测
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#传入测试文件地址
df_train=pd.read_csv('data/breast-cancer-train.csv')
df_test=pd.read_csv('data/breast-cancer-test.csv')

plt.subplot(2,2,1)
#//特征选取5
df_test_negative=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
#//绘图,良性
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
#//绘图，恶性
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
#//绘制x,y轴的说明
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
#//显示
#plt.show()

intercept=np.random.random([1])
coef=np.random.random([2])
lx=np.arange(0,12)
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c="yellow")
plt.subplot(2,2,2)
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='o',s=150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
#plt.show()

#导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

#使用前10条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))


intercept=lr.intercept_
coef=lr.coef_[0,:]
ly=(-intercept-lx*coef[0])/coef[1]

#绘图
plt.subplot(2,2,3)
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='o',s=150,c='black')
plt.xlabel('ten data')
plt.ylabel('Cell Size')
#plt.show()

lr=LogisticRegression()
#使用所有训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))


intercept=lr.intercept_
coef=lr.coef_[0,:]
ly=(-intercept-lx*coef[0])/coef[1]

#绘图
plt.subplot(2,2,4)
plt.plot(lx,ly,c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='o',s=150,c='black')
plt.xlabel('all data')
plt.ylabel('Cell Size')
plt.show()

lr=LogisticRegression()
#使用所有训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))