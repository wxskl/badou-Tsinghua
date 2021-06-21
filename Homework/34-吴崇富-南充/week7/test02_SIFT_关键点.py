#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

img = cv2.imread('./source/lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # 实例化sift
keypoints,descriptor = sift.detectAndCompute(gray,None) # 计算出图像的关键点和sift特征向量，kp是关键点的列表，des是描述子
'''
opencv3.x.官方文档里介绍cv2.drawKeypoints（）函数主要包含五个参数：
image:也就是原始图片
keypoints：从原图中获得的关键点，这也是画图时所用到的数据
outputimage：输出
color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
flags：绘图功能的标识设置，
'''
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向
img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))

cv2.imshow('sift keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()