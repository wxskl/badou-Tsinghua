# -*- encoding=UTF-8 -*-

from skimage import util
import matplotlib.pyplot as plt
import numpy as np
import random

def skimage_function():
    img_original = plt.imread("images/lenna.png")
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_original)
    img_sp = util.random_noise(img_original,mode="s&p",salt_vs_pepper=0.6)
    plt.subplot(1,2,2)
    plt.imshow(img_sp)
    plt.show()

skimage_function()

def handwrite_function():
    img_orignal = plt.imread("images/lenna.png")
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_orignal)
    salt_vs_pepper = 0.6
    h,w,c = img_orignal.shape
    img_sp = np.zeros((h,w,c))
    totalnum = h*w*c
    saltnum = 0
    peppernum = 0
    for i in range(c):
        for j in range(h):
            for k in range(w):
                # 不严谨的判断
                if random.random() <= salt_vs_pepper:
                    img_sp[j, k, i] = [0,255][random.randint(0,1)]
                    if img_sp[j, k, i] == 0:
                        saltnum+=1
                    else:
                        peppernum+=1
                else:
                    img_sp[j, k, i] = img_orignal[j, k, i]
    print((saltnum+peppernum)/totalnum)
    print(saltnum,"---",peppernum,"---",totalnum)
    plt.subplot(1,2,2)
    plt.imshow(img_sp)
    plt.show()
handwrite_function()


