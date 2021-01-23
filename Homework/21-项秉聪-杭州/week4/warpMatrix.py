# -*- encoding=UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def WarpPerspectiveMatrix(src,dst):
    # 定义应该要变换的长度
    nums = src.shape[0]  # 一般来说是4，因为图片是4个角，当然也可以增加到更多维度

    # 定义计算公式的A、B矩阵
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))

    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        # 代表着第0、2、4、6行
        A[i * 2] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2] = [B_i[0]]
        #采用i * 2 + 1，代表1、3、5、7行
        A[i * 2 + 1] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1] = [B_i[1]]
    #使用np.mat，是为了好拿到它的逆矩阵A.I
    A = np.mat(A)
    warpMatrix = A.I * B

    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape(3, 3)
    return  warpMatrix

def changeImg(warpMatrix):
    #img1 = cv2.cvtColor(cv2.imread("lenna.png"), cv2.COLOR_BGR2GRAY)
    #zeroImg = np.zeros(img1.shape[:2], dtype=np.uint8)
    img1 = cv2.imread("images/lenna.png")
    zeroImg = np.zeros(img1.shape, dtype=np.uint8)
    for i in range(len(zeroImg)):
        for j in range(len(zeroImg[i])):
            orignal = np.dot(warpMatrix, np.array([i, j, 1]).T)
            x,y,z = orignal[:]
            x = int(np.round(x/z))
            y = int(np.round(y/z))
            #处理超出范围的值
            if x >= len(zeroImg):
                x = len(zeroImg) - 1
            elif x <= 0:
                x = 0
            if y >= len(zeroImg[i]):
                y = len(zeroImg[i]) - 1
            elif y <= 0:
                y = 0
            zeroImg[i,j] = img1[x,y]

    cv2.imshow("old",img1)
    cv2.imshow("new", zeroImg)
    cv2.waitKey(0)


if __name__ == "__main__":
    #生成原始数据
    src = np.float32([[0,30],[ 512,30],[30,512],[512,512]])
    dst = np.float32([[0,0],[ 512,0],[0,512],[512,512]])
    warpMatrix = WarpPerspectiveMatrix(src,dst)
    changeImg(warpMatrix)

