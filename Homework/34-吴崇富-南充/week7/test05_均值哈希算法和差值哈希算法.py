#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC) # 这里第二个参数是(宽，高),第三个参数为插值算法
    # 转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s += gray[i,j]
    # 求平均灰度
    avg = s/64
    # 灰度大于平均值为1，相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# 差值哈希算法
def dHash(img):
    # 缩放为8*9
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，否则为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

# hash值对比
def cmpHash(hash1,hash2):
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    count = 0
    for i in range(len(hash1)):
        # 不相等则计数+1，count最终为相似度
        if hash1[i] != hash2[i]:
            count += 1
    return count

if __name__ == '__main__':
    img1 = cv2.imread('./source/lenna_color.jpg')
    img2 = cv2.imread('./source/lenna_sharp.jpg')
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    print(hash1)
    print(hash2)
    n = cmpHash(hash1,hash2)
    print('均值哈希算法相似度',n)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print(hash1)
    print(hash2)
    n = cmpHash(hash1,hash2)
    print('差值哈希算法相似度',n)

