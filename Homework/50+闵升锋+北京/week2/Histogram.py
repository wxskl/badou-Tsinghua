import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(src):
    """
    :param src: 原图像
    :return: 增强后的结果图
    """
    #定义一个数组，每个元素存储图像每个灰度级的像素个数
    hist = [0 for i in range(256)]
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            hist[src[i, j]] += 1

    return hist

def cumulate_histogram(src, dst):
    hist = histogram(src)
    # 计算灰度级累计结果
    for i in range(len(hist)):
        if i == 0:
            hist[i] = hist[i]
        else:
            hist[i] += hist[i - 1]
    # 将原图通过像素累计公式变换，得到直方图均衡化后的图像
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = (hist[src[i, j]] / (src.shape[0] * src.shape[1])) * 255

    return dst

def count_hist(hist):
    """
    :param hist: 图像的直方图，一个长度为256的数组，存放的是每个灰度级图像的像素累计个数
    :return: 计算总的个数
    """
    total = 0
    for i in hist:
        total += i
    print(total)

def plot_hist(hist):
    plt.figure()
    plt.plot(range(len(hist)), hist)
    plt.show()

def main():
    src = cv2.imread('lena.jpg', 0)
    dst =src.copy()
    #计算原图的直方图
    hist = histogram(src)
    #变换得到结果图
    result = cumulate_histogram(src, dst)
    #计算结果图直方图
    result_hist = histogram(result)
    #分别显示原图的直方图和结果图的直方图
    plot_hist(hist)
    plot_hist(result_hist)
    #显示原图和变换结果图
    cv2.imshow('src', np.hstack([src, result]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

