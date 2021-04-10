# random.gauss(means,sigma)随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
'''
高斯噪声指它的概率密度函数服从高斯分布的一类噪声。
产生原因：
1） 图像传感器在拍摄时不够明亮、 亮度不够均匀；
2） 电路各元器件自身噪声和相互影响；
3） 图像传感器长期工作， 温度过高

a. 输入参数sigma 和 mean
b. 生成高斯随机数 random.gauss(means,sigma)
d. 根据输入像素计算出输出像素  Pout = Pin + random.gauss
e. 重新将像素值放缩在[0 ~ 255]之间
f. 循环所有像素
g. 输出图像
'''

import numpy as np
import cv2
from numpy import shape
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint(a, b)生成随机整数在(a, b)之间
        # 高斯噪声图片边缘不处理，故-1

        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.imwrite('lenna_GaussianNoise.png', img1)
cv2.waitKey(0)
