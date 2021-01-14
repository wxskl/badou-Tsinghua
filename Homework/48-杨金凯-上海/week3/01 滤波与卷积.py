
'''
滤波与卷积
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Padding(image,kernels_size,stride = [1,1],padding = "same"):
    '''
    对图像进行padding
    :param image: 要padding的图像矩阵
    :param kernels_size: list 卷积核大小[h,w]
    :param stride: 卷积步长 [左右步长，上下步长]
    :param padding: padding方式
    :return: padding后的图像
    '''
    if padding == "same":
        h,w = image.shape
        p_h =max((stride[0]*(h-1)-h+kernels_size[0]),0)  # 高度方向要补的0
        p_w =max((stride[1]*(w-1)-w+kernels_size[1]),0)  # 宽度方向要补的0
        p_h_top = p_h//2                                 # 上边要补的0
        p_h_bottom = p_h-p_h_top                         # 下边要补的0
        p_w_left = p_w//2                                # 左边要补的0
        p_w_right = p_w-p_w_left                         # 右边要补的0
        # print(p_h_top,p_h_bottom,p_w_left,p_w_right)     # 输出padding方式
        padding_image = np.zeros((h+p_h, w+p_w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                    padding_image[i+p_h_top][j+p_w_left] = image[i][j] # 将原来的图像放入新图中做padding
        return padding_image
    else:
        return image


def filtering_and_convolution(image,kernels,stride,padding = "same"):
    '''
    :param image: 要卷积的图像
    :param kernels: 卷积核 列表
    :param stride: 卷积步长 [左右步长，上下步长]
    :param padding: padding方式 “same”or“valid”
    :return:
    '''
    image_h,image_w = image.shape
    kernels_h,kernels_w = np.array(kernels).shape
    # 获取卷积核的中心点
    kernels_h_core = int(kernels_h/2+0.5)-1
    kernels_w_core = int(kernels_w/2+0.5)-1
    if padding == "valid":
        # 计算卷积后的图像大小
        h = int((image_h-kernels_h)/stride[0]+1)
        w = int((image_w-kernels_w)/stride[1]+1)
        # 生成卷积后的图像
        conv_image = np.zeros((h,w),dtype=np.uint8)
        # 计算遍历起始点
        h1_start = kernels_h//2
        w1_start = kernels_w//2
        ii=-1
        for i in range(h1_start,image_h - h1_start,stride[0]):
            ii += 1
            jj = 0
            for j in range(w1_start,image_w - w1_start,stride[1]):
                sum = 0
                for x in range(kernels_h):
                    for y in range(kernels_w):
                        # print(i,j,int((i/image_h)*h),int((j/image_w)*w),  i-kernels_h_core + x,  j-kernels_w_core+y,x,y)
                        sum += int(image[i-kernels_h_core+x][j-kernels_w_core+y]*kernels[x][y])
                conv_image[ii][jj] = sum
                jj += 1
        return conv_image

    if padding == "same":
        # 对原图进行padding
        kernels_size = [kernels_h, kernels_w]
        pad_image = Padding(image,kernels_size,stride,padding="same")
        # 计算卷积后的图像大小
        h = image_h
        w = image_w
        # 生成卷积后的图像
        conv_image = np.zeros((h,w),dtype=np.uint8)
        # # 计算遍历起始点
        h1_start = kernels_h//2
        w1_start = kernels_w//2
        ii=-1
        for i in range(h1_start,image_h - h1_start,stride[0]):
            ii +=1
            jj = 0
            for j in range(w1_start,image_w - w1_start,stride[1]):
                sum = 0
                for x in range(kernels_h):
                    for y in range(kernels_w):
                        sum += int(image[i-kernels_h_core+x][j-kernels_w_core+y]*kernels[x][y])
                conv_image[ii][jj] = sum
                jj += 1
        return conv_image

def sobel_filter(image):
    h = image.shape[0]
    w = image.shape[1]
    image_new = np.zeros(image.shape, np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sx = (image[i + 1][j - 1] + 2 * image[i + 1][j] + image[i + 1][j + 1]) - \
                 (image[i - 1][j - 1] + 2 * image[i - 1][j] + image[i - 1][j + 1])
            sy = (image[i - 1][j + 1] + 2 * image[i][j + 1] + image[i + 1][j + 1]) - \
                 (image[i - 1][j - 1] + 2 * image[i][j - 1] + image[i + 1][j - 1])
            image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
            # image_new[i][j] = sy
    return image_new

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
img = cv2.imread('lenna.png',1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(331)
plt.imshow(img_gray,cmap="gray")
plt.title("原图")


sobel_Gy = [[-1,0,1],[-2,0,2],[-1,0,1]]
Average = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
Gaussian = [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]
Laplace = [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
stride=[1,1]
img_sobel_Gy = filtering_and_convolution(img_gray,sobel_Gy,stride,padding="same")
img_Average = filtering_and_convolution(img_gray,Average,stride,padding="same")
img_Gaussian = filtering_and_convolution(img_gray,Gaussian,stride,padding="same")
img_Laplace = filtering_and_convolution(img_gray,Laplace,stride,padding="same")
plt.subplot(332)
plt.imshow(img_sobel_Gy,cmap = "gray")
plt.title("sobel_Gy")
plt.subplot(333)
plt.imshow(img_Average,cmap = "gray")
plt.title("Average")
plt.subplot(334)
plt.imshow(img_Gaussian,cmap = "gray")
plt.title("Gaussian")
plt.subplot(335)
plt.imshow(img_Laplace,cmap = "gray")
plt.title("Laplace")
plt.show()

