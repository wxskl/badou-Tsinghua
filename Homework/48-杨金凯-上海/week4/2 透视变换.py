import cv2
import numpy as np
import matplotlib.pyplot as plt
################################################################################################
#           根据四对点计算warpMatrix矩阵
################################################################################################
def get_warpMatrix(src,dst):

    # 原图像的四个点
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]

    # 变换后图像的四个点
    x0_, y0_ = dst[0]
    x1_, y1_ = dst[1]
    x2_, y2_ = dst[2]
    x3_, y3_ = dst[3]
    # 构造方程系数矩阵
    A = np.zeros([8, 8])
    B = np.zeros([8, 1])
    A[0, :] = [x0, y0, 1, 0, 0, 0, -x0 * x0_, -y0 * x0_]
    A[1, :] = [0, 0, 0, x0, y0, 1, -x0 * y0_, -y0 * y0_]
    A[2, :] = [x1, y1, 1, 0, 0, 0, -x1 * x1_, -y1 * x1_]
    A[3, :] = [0, 0, 0, x1, y1, 1, -x1 * y1_, -y1 * y1_]
    A[4, :] = [x2, y2, 1, 0, 0, 0, -x2 * x2_, -y2 * x2_]
    A[5, :] = [0, 0, 0, x2, y2, 1, -x2 * y2_, -y2 * y2_]
    A[6, :] = [x3, y3, 1, 0, 0, 0, -x3 * x3_, -y3 * x3_]
    A[7, :] = [0, 0, 0, x3, y3, 1, -x3 * y3_, -y3 * y3_]
    B[0, 0] = x0_
    B[1, 0] = y0_
    B[2, 0] = x1_
    B[3, 0] = y1_
    B[4, 0] = x2_
    B[5, 0] = y2_
    B[6, 0] = x3_
    B[7, 0] = y3_
    # 解方程组求出a11, a12, a13, a21, a22, a23, a31, a32
    # 求矩阵A的逆矩阵
    A_inv = np.linalg.inv(A)  # 矩阵求逆
    # warpMatrix=A的逆矩阵乘B
    warpMatrix = A_inv.dot(B)  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warpMatrix = np.append(warpMatrix, [1])  # 构造变换矩阵，添加 a33   8x1=>9x1
    warpMatrix = np.reshape(warpMatrix, [3, 3])  # 9x1 => 3x3
    print(warpMatrix)
    return warpMatrix
################################################################################################
#           根据warpMatrix矩阵计算变换后的图像 new_image在原图向image中的对应位置  重点理解
################################################################################################
def warpPerspective(image,warpMatrix):
    '''
    :param image: 要变换的图像
    :param warpMatrix: 变换矩阵
    :return:
    '''
    a11 = warpMatrix[0, 0]
    a12 = warpMatrix[0, 1]
    a13 = warpMatrix[0, 2]
    a21 = warpMatrix[1, 0]
    a22 = warpMatrix[1, 1]
    a23 = warpMatrix[1, 2]
    a31 = warpMatrix[2, 0]
    a32 = warpMatrix[2, 1]
    a33 = warpMatrix[2, 2]

    H,W,C = image.shape
    new_image = np.zeros([H,W,C],np.uint8)
    # 遍历的是新图像的行列而不是原图像的行列
    for Y in range(H):      # 行
        for X in range(W):  # 列
            # 反向求变换后的图像 new_image在原图向image中的对应位置，这样方便给出新图的大小  要特别注意 重点理解---使用手推公式计算
            x = ((a22-a32*Y)*(a33*X-a13)-(a12-a32*X)*(a33*Y-a23))/((a22-a32*Y)*(a11-a31*X)-(a12-a32*X)*(a21-a31*Y))
            y = ((a21-a31*Y)*(a33*X-a13)-(a11-a31*X)*(a33*Y-a23))/((a21-a31*Y)*(a12-a32*X)-(a11-a31*X)*(a22-a32*Y))
            # 像素坐标应为整型
            x = int(np.round(x))
            y = int(np.round(y))
            # 如果对应到原图时，超出原图的范围则，超出的位置直接等于边界位置（简单粗暴）
            if x < 0:
                x = 0
            if x > W - 1:
                x = W - 1
            if y < 0:
                y = 0
            if y > H-1:
                y = H-1
            # 行列对应于遍历的行列 要特别注意 重点理解
            new_image[Y,X] = image[y,x]
    return new_image

def warpPerspective1(image,warpMatrix):
    '''
    :param image: 要变换的图像
    :param warpMatrix: 变换矩阵
    :return:
    '''
    warpMatrix_inv = np.linalg.inv(warpMatrix)  # 求矩阵warpMatrix逆矩阵

    H,W,C = image.shape
    new_image = np.zeros([H,W,C],np.uint8)
    # 遍历的是新图像的行列而不是原图像的行列
    for Y in range(H):      # 行
        for X in range(W):  # 列
            # 反向求变换后的图像 new_image在原图向image中的对应位置，这样方便给出新图的大小  要特别注意 重点理解---使用矩阵运算计算
            x,y,_ = warpMatrix_inv.dot(np.array([X,Y,1]))
            # 像素坐标应为整型
            x = int(np.round(x))
            y = int(np.round(y))
            # 如果对应到原图时，超出原图的范围则，超出的位置直接等于边界位置（简单粗暴）
            if x < 0:
                x = 0
            if x > W - 1:
                x = W - 1
            if y < 0:
                y = 0
            if y > H-1:
                y = H-1
            # 行列对应于遍历的行列 要特别注意 重点理解
            new_image[Y,X] = image[y,x]
    return new_image

# 原图像的四个点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# 对应变换后的四个点
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 计算变换矩阵
warpMatrix = get_warpMatrix(src,dst)

image = cv2.imread("photo1.jpg", 1)
# 根据变换矩阵进行图像变换
new_image = warpPerspective(image, warpMatrix)
cv2.imshow("src",image)
cv2.waitKey(0)

cv2.imshow("dst",new_image)
cv2.waitKey(0)













