#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

'''
双线性插入缩放
python implementation of bilinear interpolation
'''

def s2d(img, theSize):
    img_new = np.zeros((theSize[0], theSize[1], 3), dtype=np.uint8)
    sh, sw, sc = img.shape
    dh, dw, dc = img_new.shape
    scale_y = dh / sh
    scale_x = dw / sw
    for i in range(3):
        for sy in range(dh):
            for sx in range(dw):
                # 遍历新图元素坐标获取初始的原图浮点坐标
                x = (sx + 0.5) / scale_x - 0.5
                y = (sy + 0.5) / scale_y - 0.5
                # 根据浮点坐标，获取用于计算的两个原图坐标值
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))
                x2 = min(x1 + 1, sw - 1)
                y2 = min(y1 + 1, sh - 1)
                # 根据公式计算要填充的像素值
                img_new[sy, sx, i] = int((y2 - y) * ((x2 - x) * img[y1, x1, i] + (x - x1) * img[y1, x2, i]) + (y - y1) * ((x2 - x) * img[y2, x1, i] + (x - x1) * img[y2, x2, i]))

    return img_new


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img1 = s2d(img, (512, 700))
    cv2.imshow('img', img)
    cv2.imshow('new', img1)
    cv2.waitKey(0)
