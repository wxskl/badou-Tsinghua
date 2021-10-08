from skimage.color import rgb2gray
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import matplotlib.image as im  # im 用于读取图片
import numpy as np
import cv2

def grayAndBinary(img):
    '''
    1、三通道彩色图转灰度图
    2、灰度图二值化
    各通道的权重：
    Y = 0.2125 R + 0.7154 G + 0.0721 B
    :param img:
    :return:
    '''
    print(img.shape)
    # 设置2行，2列的图像网格,img填充第一个单元格
    plt.subplot(221)
    plt.imshow(img)

    # 彩色图片灰度化处理
    gray_img = rgb2gray(img)
    # gray_img填充第二个单元格
    plt.subplot(222)
    plt.imshow(gray_img)

    # rgb2grey = rgb2gray
    grey_img = rgb2grey(img)
    plt.subplot(223)
    plt.imshow(grey_img)

    (h, w) = gray_img.shape
    two_value_img = np.zeros(gray_img.shape, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            ori_px = gray_img[i, j]
            # 灰度化处理后的图像像素值已经转为[0,1]浮点数
            if ori_px <= 0.5:
                two_value_img[i, j] = 0
            else:
                two_value_img[i, j] = 1

    plt.subplot(224)
    plt.imshow(two_value_img)
    plt.show()


def bilinear_interpolation(img, out_dim):
    '''
    双线性插值
    :param img: 输入图片
    :param out_dim: 输出图片纬度[h,w]
    :return:
    '''
    src_h, src_w, channel = img.shape
    dst_img = np.zeros((out_dim[0], out_dim[1], 3), dtype=np.uint8)
    print(dst_img.shape)
    (h, w, c) = dst_img.shape

    # 计算原始图片与目标图片的比例
    scale_y = float(src_h) / h
    scale_x = float(src_w) / w
    for n in range(3):
        for dst_y in range(h):
            for dst_x in range(w):
                # 计算目标做标在原图像的位置
                src_x = scale_x * (dst_x + 0.5) - 0.5
                src_y = scale_y * (dst_y + 0.5) - 0.5

                # 目标在原图上最邻近的四个点的坐标
                src_x_1 = int(np.floor(src_x))
                src_y_1 = int(np.floor(src_y))
                src_x_2 = min(src_x_1 + 1, src_w - 1)
                src_y_2 = min(src_y_1 + 1, src_h - 1)

                # 双线性插值计算
                # X方向进行两次插值，得到temp0(原始图像中坐标[src_y_1, src_x_1]和坐标[src_y_1, src_x_2])和temp1(原始图像中坐标[src_y_2, src_x_1]和坐标[src_y_2, src_x_2])两个像素值
                temp0 = (src_x_2 - src_x) * img[src_y_1, src_x_1, n] + (src_x - src_x_1) * img[src_y_1, src_x_2, n]
                temp1 = (src_x_2 - src_x) * img[src_y_2, src_x_1, n] + (src_x - src_x_1) * img[src_y_2, src_x_2, n]
                # Y方向进行插值，得到目标图像坐标[dst_y, dst_x]的像素值
                dst_img[dst_y, dst_x, n] = int((src_y_2 - src_y) * temp0 + (src_y - src_y_1) * (temp1))

    return dst_img


def hisogram_equalization(img):
    '''
    直方图均衡化
    :param img: 原始三通道图片
    :return: 均衡化处理后的图片
    '''
    cv2.imshow('src img', img)
    # 分割三通道图片
    (b, g, r) = cv2.split(img)
    bh = _equalization(b)
    gh = _equalization(g)
    rh = _equalization(r)

    # bh = cv2.equalizeHist(b)
    # gh = cv2.equalizeHist(g)
    # rh = cv2.equalizeHist(r)

    dst_img = cv2.merge((bh, gh, rh))
    print(dst_img.shape)
    plt.hist(rh.flatten(), bins=256, density=1)
    plt.show()
    return dst_img

def _equalization(img):
    '''
    对输入图片进行均衡化处理
    :param img: 单通道图片
    :return: 均衡化之后的单通道图片
    '''
    (h, w) = img.shape
    img_vector = img.flatten()
    ni = [0] * 256
    for i in img_vector:
        ni[i] += 1
    pi = [ni[j] / (h * w) for j in range(256)]

    sum_pi = [0] * 256
    # 初始累加第一个值
    sum_pi[0] = pi[0]
    # 从1开始遍历累加
    for i in range(1, 256):
        sum_pi[i] = sum_pi[i - 1] + pi[i]

    dst_img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            dst_img[y, x] = sum_pi[img[y, x]] * 256 - 1

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # # 灰度处理&二值化处理
    # grayAndBinary(img)
    # # 双线性插值
    dst = bilinear_interpolation(img, [1080, 1080])
    # dst = hisogram_equalization(img)
    cv2.imshow('histogram equliazation img',dst )
    # cv2.imshow('bilinear interpolation', dst)
    # cv2.imwrite('lenna.1080.1080.png', dst)
    # cv2.waitKey()


    # 直方图均衡化测试数据
    # arr = np.array([[1, 3, 9, 9, 8], [2, 1, 3, 7, 3], [3, 6, 0, 6, 4], [6, 8, 2, 0, 5], [2, 9, 2, 6, 0]])
    # print(_equalization(arr))
