#!/usr/bin/env python
# encoding=gbk

'''
直方图均衡化；cv2里有cv2.equalization函数可直接将图像做均衡化处理
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def Histogram(img):
    '''
    1.依次扫描原始图像的每一个像素，计算出图像的灰度直方图H
    2.计算灰度直方图的累加直方图
    :return:原图的灰度直方图
    '''
    h, w = img.shape
    hist0 = [0 for x in range(256)]
    for i in range(h):
        for j in range(w):
            hist0[img[i, j]] = hist0[img[i, j]] + 1
        pass
    return hist0

def Histogram_equalization(img, hist):
    '''
    根据累加直方图和直方图均衡化原理得到的输入与输出之间的映射关系将图像进行变换
    :param img: 原图
    :param hist: 原图的累加直方图
    :return: 均衡化之后的图像
    '''
    h, w = img.shape
    # equalization = [[0 for x in range(h)] for x in range(w)]
    # equalization = np.array(equalization)
    # print(type(equalization))
    for i in range(h):
        for j in range(w):
            sumPi = 0
            p = img[i, j]
            for k in range(p):
                sumPi = sumPi + hist[k]
                pass
            img[i, j] = int(sumPi*256/(h*w))-1
            sumPi = 0
            pass
        pass
    # np.array(equalization)
    # print(type(equalization))
    return img

# 灰度图像的均衡化
img = cv2.imread('lenna.png')
# print(type(img))

# hight, width, channels = img.shape
# channels = 1
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure('gray1') # 图像窗口名字
plt.imshow(img)
plt.show()
# print(img)
# cv2.imshow('gray1', img)
# cv2.waitKey(0)

hist0 = Histogram(img)

equalization = Histogram_equalization(img, hist0)

plt.figure('gray') # 图像窗口名字
plt.imshow(equalization)
plt.show()
cv2.imshow('gray', equalization)
cv2.imwrite('Hist_Equ_Graylenna.png', equalization)  # 保存图像
print(type(equalization))
print(equalization.shape)


# 彩色图像的均衡化
img1 = cv2.imread('lenna.png')
hist0 = Histogram(img1[:, :, 0])
hist1 = Histogram(img1[:, :, 1])
hist2 = Histogram(img1[:, :, 2])

equalization0 = Histogram_equalization(img1[:, :, 0], hist0)
equalization1 = Histogram_equalization(img1[:, :, 1], hist1)
equalization2 = Histogram_equalization(img1[:, :, 2], hist2)
equalization = cv2.merge((equalization0, equalization1, equalization2))
equalization = cv2.cvtColor(equalization, cv2.COLOR_BGR2RGB)

cv2.imshow('gray', equalization)
cv2.imwrite('Hist_Equ_RGBlenna.png', equalization)  # 保存图像



# # 灰度图像的直方图均衡化
img2 = cv2.imread('lenna.png')  # 读取原图
gray = cv2.cvtColor(img2, cv2.COLOR_BAYER_BG2GRAY)  # cv2默认读取BGR格式转为gray图

# 直方图均衡化
dst = cv2.equalizeHist(gray)
# 画出直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
'''
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
此函数用于计算图像直方图：
    imaes:输入的图像
    channels:选择图像的通道
    mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
    histSize:使用多少个bin(柱子)，一般为256
    ranges:像素值的范围，一般为[0,255]表示0~255
'''
plt.figure()
plt.hist(dst.ravel(), 256)  # numpy.ravel()将多维数组转为一维数组
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# 拼接数组：np.vstack():在竖直方向上堆叠 np.hstack():在水平方向上平铺
cv2.waitKey(0)

'''
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''

