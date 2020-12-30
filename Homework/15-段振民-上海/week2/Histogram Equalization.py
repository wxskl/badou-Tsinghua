#!/usr/bin/env python
# encoding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#原灰度直方图
plt.figure()
plt.hist(gray.ravel(), 256)
#均衡化灰度直方图
plt.figure()
img_1 = cv2.equalizeHist(gray)
plt.hist(img_1.ravel(), 256)
#原彩色直方图
image = cv2.imread("lenna.png")
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (i,color) in enumerate(colors):
    hist = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
#彩图均衡化对比
img = cv2.imread('lenna.png')
b,g,r = cv2.split(img)
ab = cv2.equalizeHist(b)
ag = cv2.equalizeHist(g)
ar = cv2.equalizeHist(r)
equals_img = cv2.merge((ab,ag,ar))
cv2.imshow("123", np.hstack([img, equals_img]))
plt.show()
cv2.waitKey(0)





