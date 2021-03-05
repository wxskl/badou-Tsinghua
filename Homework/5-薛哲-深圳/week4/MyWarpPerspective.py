'''
1.利用原图需要变换的物体的四个顶点坐标和变换后的四个顶点坐标求出变换矩阵warpMatrix
2.根据透视变换公式和已求出的warpMatrix矩阵求出新图像在原图中的位置和像素值
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_warpMatrix(src, dst):
    '''
    利用原图需要变换的物体的四个顶点坐标和变换后的四个顶点坐标求出变换矩阵warpMatrix
    A * warpMatrix = B
    :param src: 原图需要变换物体的四个顶点
    :param dst: 新图对应的四个顶点
    :return: warpMatrix
    '''

    B = np.zeros((8, 1))
    for i in range(4):
        # 先做一个2*8的矩阵
        A1 = np.zeros((2, 8))

        A1[0, 0] = src[i,0]
        A1[0, 1] = src[i,1]
        A1[0, 2] = 1
        A1[0, 6] = -src[i, 0] * dst[i, 0]
        A1[0, 7] = -src[i, 1] * dst[i, 0]
        B[2*i] = dst[i, 0]
        A1[1, 3] = src[i, 0]
        A1[1, 4] = src[i, 1]
        A1[1, 5] = 1
        A1[1, 6] = -src[i, 0] * dst[i, 1]
        A1[1, 7] = -src[i, 1] * dst[i, 1]
        B[2*i+1] = dst[i, 1]

        if i == 0:
            A = A1
        else:
            A = np.vstack([A, A1]) # 连接四个2*8的矩阵
            pass
        pass
    A = np.mat(A)
    # print(A)
    warpMatrix = A.I*B
    # print(warpMatrix)
    # 已求出a11,a12,a13,a21,a22,a23,a31,a32  再插入a33 = 1
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def get_result(warpMatrix, dst, img):
    '''

    :param warpMatrix:
    :param dst:
    :return:
    '''
    w = int(dst[1, 0] - dst[0, 0])+1  # 337
    h = int(dst[2, 1] - dst[0, 1])+1  # 488

    # 用opencv处理图像时，可以发现获得的矩阵类型都是uint8 无符8位整型（0-255）, uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
    result = np.zeros((h, w, 3), np.uint8)

    # 数组与数组运算，矩阵与矩阵运算，最好统一数据类型，实测数组与矩阵相乘结果会不一样
    # W = np.linalg.inv(warpMatrix)
    W = np.mat(warpMatrix)
    W = W.I
    for i in range(h):
        for j in range(w):

            XY1 = np.array([[j], [i], [1]])
            # 关于j,i 的位置问题，由给定的dst坐标排列决定，由src和dst算出的变换矩阵已经定好位置，所以在反求原图位置的时候需注意
            XY1 = np.mat(XY1)
            # XY1 = XY1.T    一维数组无法进行转置
            # print(XY1)
            x, y, _ = W.dot(XY1)

            # 验证算出来的坐标是否和src里的相对应
            if i ==h-1 and j == 0:
                print(x, y)

            # 算出的索引超出原图大小时，令其等于边界
            if y >=960:
                y = 959
                pass
            if y < 0:
                y = 0
                pass
            if x >=540:
                x = 539
                pass
            if x < 0:
                x = 0
                pass
            # print(int(xy1[0]))
            # try:
            #     result[i, j, 0] = img[int(xy1[0]), int(xy1[1]), 0]
            #     result[i, j, 1] = img[int(xy1[0]), int(xy1[1]), 1]
            #     result[i, j, 2] = img[int(xy1[0]), int(xy1[1]), 2]
            # except Exception as msg:
            #     print(xy1[0], xy1[1])

            # result[i, j] = img[int(y), int(x)]

            result[i, j, 0] = img[int(y), int(x), 0]
            result[i, j, 1] = img[int(y), int(x), 1]
            result[i, j, 2] = img[int(y), int(x), 2]


    return result

if __name__ == '__main__':

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = get_warpMatrix(src, dst)
    img_src = cv2.imread('photo1.jpg', 1)
    print(img_src.shape)
    img = np.copy(img_src)
    res = get_result(warpMatrix, dst, img)
    cv2.imshow('img_src', img_src)
    cv2.imshow('res', res)
    cv2.imwrite('warp_photo.jpg', res)
    cv2.waitKey(0)
