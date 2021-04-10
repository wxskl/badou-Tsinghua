#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def aHash(img,width=8,high=8):
    '''
    均值哈希算法
    :param img:图像数据
    :param width:图像缩放的宽度
    :param high:图像缩放的高度
    :return:均值哈希序列
    '''
    # 缩放图像
    img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(high):
        for j in range(width):
            s += gray[i,j]
    # 求平均灰度
    avg = s/(width*high)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(high):
        for j in range(width):
            if gray[i,j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def dHash(img,width=9,high=8):
    '''
    差值哈希算法
    :param img: 图像数据
    :param width: 图像缩放的宽度
    :param high: 图像缩放的高度
    :return: 差值哈希序列
    '''
    # 缩放图像
    img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，反之置为0，生成差值哈希序列（string）
    for i in range(high):
        for j in range(width-1):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

def cmp_hash(hash1,hash2):
    '''
    Hash值对比
    :param hash1:哈希序列1
    :param hash2:哈希序列2
    :return:相似度
    '''
    count = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则计数+1
        if hash1[i] != hash2[i]:
            count += 1
    return 1-count/len(hash2)

def pHash(img,width=64,high=64):
    '''
    感知哈希算法
    :param img:图像数据
    :param width:图像缩放后的宽度
    :param high:图像缩放后的高度
    :return:图像感知哈希序列
    '''
    # 加载并调整图片为32x32灰度图片,不对啊，用的是64*64
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(width,high),interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    h,w = img.shape[:2]
    vis0 = np.zeros((h,w),np.float32) # 数据类型设置为np.float32是为了方便做DCT转换吗?
    vis0[:h,:w] = img # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(32,32)

    # 把二维list变成一维list
    img_list = vis1.flatten()

    # 计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['1' if i > avg else '0' for i in img_list]

    # 得到哈希值 (注意这行代码)
    return ''.join(['%x' %int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])

def hamming_dist(s1,s2):
    return 1 - sum([ch1!=ch2 for ch1,ch2 in zip(s1,s2)])*1./(32*32/4)

def concat_info(type_str,score,time):
    temp = '%s相似度，%.2f%% -----time=%.4f ms' %(type_str,score*100,time)
    print(temp)
    return temp

def test_diff_hash(img1_path,img2_path,loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    start_time = time.time()

    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1,hash2)
    print('>>> 执行%s次耗费的时间为%.4f s.' %(loops,time.time()-start_time))

def test_aHash(img1,img2):
    time1 = time.time()
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmp_hash(hash1,hash2)
    return concat_info('均值哈希算法',n,time.time()-time1) + '\n'

def test_dHash(img1,img2):
    time1 = time.time()
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmp_hash(hash1,hash2)
    return concat_info('差值哈希算法',n,time.time()-time1) + '\n'

def test_pHash(img1,img2):
    time1 = time.time()
    hash1 = pHash(img1)
    hash2 = pHash(img2)
    n = hamming_dist(hash1,hash2)
    return concat_info('感知哈希算法',n,time.time()-time1) + '\n'

def deal(img1_path,img2_path):
    info = ''
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # 计算图像哈希相似度
    info += test_aHash(img1,img2)
    info += test_dHash(img1,img2)
    info += test_pHash(img1,img2)
    return info

if __name__ == '__main__':
    # 测试算法效率
    test_diff_hash('./source/lenna.png','./source/lenna.png')
    test_diff_hash('./source/lenna.png','./source/lenna_light.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_resize.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_contrast.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_sharp.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_blur.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_color.jpg')
    test_diff_hash('./source/lenna.png','./source/lenna_rotate.jpg')

    # 测试算法的精度(以base和light为例)
    deal('./source/lenna.png','./source/lenna_light.jpg')

'''
• aHash：均值哈希。速度比较快，但是有时不太精确。
• pHash：感知哈希。精确度较高，但是速度方面较差一些。
• dHash：差值哈希。精确度较高，且速度也非常快。
'''