import numpy as np
import cv2
# import matplotlib.pyplot as plt

#最近邻插值缩放算法
# def plot_img(img):

def show_image(img,str):
    #显示图片
    """
    :param img: 要显示的图片
    :param str: 显示窗口名字
    :return:
    """
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def magnify(src, dst, transform_height, transform_width):
    """
    :param src: 需要被缩放的原图
    :param dst: 一个形状为缩放后大小的全零矩阵
    :param transform_height: 缩放后的高
    :param transform_width: 缩放后的宽
    :return: 缩放后的填充矩阵
    """
    scale_h = (src.shape[0]-1) / (transform_height-1)
    scale_w = (src.shape[1]-1) / (transform_width-1)
    for i in range(transform_height):
        n = cast(i * scale_h)
        for j in range(transform_width):
            m = cast(j * scale_w)

            dst[i, j] = src[n, m]

    return dst

def cast(number):
    #实现获取结果图坐标映射回原图的坐标的函数
    """
    :param number: number
    :return: 取整后的数
    """
    if number - int(number) < 0.5:
        number = int(number)
    else:
        number = int(number) + 1
    return number


def main():
    transform_height = input()
    transform_width = input()
    transform_height = int(transform_height)
    transform_width = int(transform_width)
    src = cv2.imread('E:/image/lena.jpg',0)
    dst = np.zeros((transform_height, transform_width),np.uint8)
    dst = magnify(src, dst, transform_height, transform_width)

#不加上这两行代码，结果显示不出来，显示的是一张白图，加上后可以正常显示
    # cv2.imwrite('./3.jpg', dst)
    # dst = cv2.imread('./3.jpg')

    show_image(dst,'dst1')
    show_image(src, 'src')


if __name__ == '__main__':
    main()