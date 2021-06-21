#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

def gray_to_hist():
    # �Ҷ�ͼ��ֱ��ͼ���⻯
    img = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = cv2.equalizeHist(gray)

    # ֱ��ͼ
    hist = cv2.calcHist([dst],[0],None,[256],[0,256])

    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()

    cv2.imshow("Histogram Result", np.hstack([gray, dst]))
    cv2.waitKey(0)

def color_to_hist():
    img = cv2.imread("lenna.png", 1)
    cv2.imshow("src", img)

    # ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(cv2.split(img), colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

    # �ϲ�ÿһ��ͨ��
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", result)

    cv2.waitKey(0)

if __name__ == '__main__':
    gray_to_hist()