import numpy as np
import cv2
import random


def sp(img, perc, mean, sigma):
    noise_img = img.copy()
    noise_num = int(perc * img.shape[0] * img.shape[1])  # 根据百分比计算生成的噪声点数
    for i in range(noise_num):
        x = random.randint(0, img.shape[0] - 1)  # 生成随机的x位置
        y = random.randint(0, img.shape[1] - 1)  # 生成随机的y位置
        noise_img[x, y] = noise_img[x, y] + random.gauss(mean, sigma) #在该随机位置加上高斯随机数
        for j in range(0, 3): #每个通道各自判断
            if noise_img[x, y][j] < 0: #如果小于0就等于0
                noise_img[x, y][j] = 0
            elif noise_img[x, y][j] > 255:  #如果大于255就等于255
                noise_img[x, y][j] = 255
    return noise_img


img = cv2.imread("cat.png")
noise_img = sp(img, 0.8, 2, 4)

cv2.imshow("noise_sp", noise_img)
cv2.imshow("source", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
