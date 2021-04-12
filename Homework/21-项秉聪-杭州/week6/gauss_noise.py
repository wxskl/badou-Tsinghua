# -*- encoding=UTF-8 -*-

from skimage import util
import cv2
import numpy as np
import random

def skimage_function():
    img_original = cv2.imread("images/lenna.png")
    cv2.imshow("img_original1",img_original)
    img_gauss = util.random_noise(img_original,mode='gaussian')
    cv2.imshow("img_gauss1",img_gauss)

skimage_function()

def handwrite_function():
    img_original = cv2.imread("images/lenna.png")
    cv2.imshow("img_original2", img_original)
    h,w,c = img_original.shape
    img_gauss = np.zeros((h,w,c))
    for i in range(c):
        for j in range(h):
            for k in range(w):
                img_gauss[j,k,i] = img_original[j,k,i] + random.gauss(0,29.9)
                if img_gauss[j,k,i] < 0:
                    img_gauss[j, k, i] = 0
                elif img_gauss[j,k,i] > 255:
                    img_gauss[j, k, i] = 255

    cv2.imshow("img_gauss2", img_gauss.astype("uint8"))

handwrite_function()
cv2.waitKey(0)
cv2.destroyAllWindows()