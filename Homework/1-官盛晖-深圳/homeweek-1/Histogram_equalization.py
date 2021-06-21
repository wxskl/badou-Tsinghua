#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
# cv2.imread(path, n)   n: 1 for bgr (default); 0 for gray; -1 for alpha
# 单一通道灰度值均衡化 the equalization of single channel gray values
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR values to gray values

dst = cv2.equalizeHist(gray)
# equalizeHist(src, dst)
# src：图像矩阵(单通道图像)
# dst：默认即可
# hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
hist = cv2.calcHist([dst], [0], None, [256], [0, 255])
"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]])
imaes:输入的图像
channels:选择图像的通道
mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
histSize:使用多少个bin(柱子)，一般为256
ranges:像素值的范围，一般为[0,255]表示0~255
注意，除了mask，其他四个参数都要带[]号。
"""

plt.figure()
plt.hist(dst.ravel(), 256)
# numpy.ravel()把多维数组降为一维数组
# numpy.flatten() 也用于降为一维数组，但返回拷贝；不直接修改原数组
plt.show()

cv2.imshow("Histogram equalization", np.hstack(gray, dst)) # show two img: gray and dst
cv2.waitKey(0)

# split the rgb to 3 channels
b, g, r = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# merge the 3 channels matrix
result = cv2.merge(bH, gH, rH)
# show the dst img
cv2.imshow("equalization of bgr", result)
cv2.waitKey(0)


