import numpy as np
import cv2
import random


def sp(img, perc):
    noise_img = img
    noise_num = int(perc * img.shape[0] * img.shape[1])  # 根据百分比计算生成的噪声点数
    for i in range(noise_num):
        x = random.randint(0, img.shape[0] - 1)  # 生成随机的x位置
        y = random.randint(0, img.shape[1] - 1)  # 生成随机的y位置
        if random.random() <= 0.5:  # 随机指定是椒噪声还是盐噪声，各占一半
            noise_img[x, y] = 0
        else:
            noise_img[x, y] = 255
    return noise_img


img = cv2.imread("cat.png")
noise_img = sp(img, 0.01)

cv2.imshow("noise_sp", noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
