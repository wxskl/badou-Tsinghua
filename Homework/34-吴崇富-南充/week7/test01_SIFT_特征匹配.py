#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
def drawMatchchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch):
    '''
    函数功能：查询图像与目标图像的关键点连线
    '''
    h1,w1 = img1_gray.shape[:2]
    h2,w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1,h2),w1+w2,3),np.uint8)
    vis[:h1,:w1] = img1_gray
    vis[:h2,w1:w1+w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch] # kpp.queryIdx查询图像中描述符的索引
    p2 = [kpp.trainIdx for kpp in goodMatch] # kpp.trainIdx目标图像中描述符的索引

    # https://blog.csdn.net/dcrmg/article/details/78817988
    post1 = np.int32([kp1[pp].pt for pp in p1]) # 查询图像中关键点的像素坐标
    post2 = np.int32([kp2[pp].pt for pp in p2])+(w1,0) # 目标图像中关键点的像素坐标+(w1,0),为什么要加(w1,0),因为查询图像与目标图像拼接在了同一张图像中。

    for (x1,y1),(x2,y2) in zip(post1,post2):
        '''
        cv2.line()函数,这个函数是opencv中用于在图像中划线的函数
        cv2.line(plot,(0,y),(int(h * mul),y),(255,0,0),w)
        第一个参数 img：要划的线所在的图像;
　　    第二个参数 pt1：直线起点
　　    第三个参数 pt2：直线终点
　　    第四个参数 color：直线的颜色
　　    第五个参数 thickness=1：线条粗细
        '''
        cv2.line(vis,(x1,y1),(x2,y2),(0,0,255))

    cv2.namedWindow('match',cv2.WINDOW_NORMAL) # 窗口大小可以改变
    cv2.imshow('match',vis)

img1_gray = cv2.imread('iphone1.png')
img2_gray = cv2.imread('iphone2.png')

# sift = cv2.SIFT() # 报错：module 'cv2.cv2' has no attribute 'SIFT'
# sift = cv2.SURF() # 报错：module 'cv2.cv2' has no attribute 'SURF'
sift = cv2.xfeatures2d.SIFT_create() # 实例化sift

kp1,des1 = sift.detectAndCompute(img1_gray,None) # 计算出图像的关键点和sift特征向量，kp是关键点的列表，des是描述子
kp2,des2 = sift.detectAndCompute(img2_gray,None) 

# BFmatcher with default parms
# https://www.pianshen.com/article/16301505836/
# https://blog.csdn.net/yukinoai/article/details/89055860
bf = cv2.BFMatcher(cv2.NORM_L2)  # 创建BFMatcher对象
matches = bf.knnMatch(des1,des2,k=2) # 用knn法特征点匹配， k=2 定义基准图像上的一个点会在另一幅图像上有2个匹配结果。

# https://blog.csdn.net/dcrmg/article/details/78817988
goodMatch = []
for m,n in matches:
    # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的0.5，
    # 基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点。当然并不能保证goodMatch保留的就是最优匹配
    if m.distance < 0.5*n.distance: # m.distance，n.distance描述符之间的距离。 越低越好
        goodMatch.append(m)

drawMatchchesKnn_cv2(img1_gray,kp1,img2_gray,kp2,goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllwindows()