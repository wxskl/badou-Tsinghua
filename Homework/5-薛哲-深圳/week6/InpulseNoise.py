'''
椒盐噪声又称脉冲噪声，为随机出现的白点或者黑点，0(椒) 255(盐)
如果通信时出错，部分像素的值在传输时丢失就会产生这种噪声。
椒盐噪声的成因可能是影像讯号受到突如其来的强烈干扰而产生等。 例如失效的感应器导致像素值
为最小值， 饱和的感应器导致像素值为最大值。

给一副数字图像加上椒盐噪声的处理顺序：
    1.指定信噪比 SNR（信号和噪声所占比例） ， 其取值范围在[0, 1]之间
    2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
    3.随机获取要加噪的每个像素位置P（i, j）
    4.指定像素值为255或者0。
    5.重复3, 4两个步骤完成所有NP个像素的加噪
'''

import numpy as np
import cv2
from numpy import shape
import random


def fun1(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0

        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = fun1(img, 0.2)
# 在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
# cv2.imwrite('lenna_PepperandSalt.png',img1)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_PepperandSalt', img1)
cv2.imwrite('lenna_PepperandSalt.png', img1)
cv2.waitKey(0)
