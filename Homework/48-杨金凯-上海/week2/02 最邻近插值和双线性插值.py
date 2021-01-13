
import cv2
import numpy as np
import math

scr_img = cv2.imread("lenna.png", 1)
print(scr_img.shape)
cv2.imshow("scr_img", scr_img)
cv2.waitKey(0)
# print(scr_img[0][0])
def nearest_neighbor_interpolation(scr_img,dstHeight,dstWidth):
    if(len(scr_img.shape)>2):# 图像不是灰度图像
        srcHeight, srcWidth, channels = scr_img.shape  # 获取原图的形状
        # 创建新图
        dst_img = np.zeros((dstHeight,dstWidth,channels),dtype = np.uint8)
        # 遍历新图,计算新图像素坐标在原图上的映射坐标
        for dstH in range(dstHeight):
            for dstW in range(dstWidth):
                scrH = round((dstH)*(srcHeight/dstHeight))
                scrW = round((dstW)*(srcWidth/dstWidth))
                dst_img[dstH][dstW] = scr_img[scrH][scrW]
        return  dst_img
    else:#图像是灰度图像
        srcHeight, srcWidth = scr_img.shape  # 获取原图的形状
        dst_img = np.zeros((dstHeight, dstWidth), dtype=np.uint8)
        # 遍历新图,计算新图像素坐标在原图上的映射坐标
        for dstH in range(dstHeight):
            for dstW in range(dstWidth):
                scrH = round((dstH) * (srcHeight / dstHeight))
                scrW = round((dstW) * (srcWidth / dstWidth))
                dst_img[dstH][dstW] = scr_img[scrH][scrW]
        return dst_img
dstHeight,dstWidth = 1000,1000
dst_img = nearest_neighbor_interpolation(scr_img,dstHeight,dstWidth)
cv2.imshow("dst_img",dst_img)
cv2.waitKey(0)


## ---------------------------------双线性插值---------------------------------------
def bilinear_interpolation(scr_img,dstHeight,dstWidth):
    if(len(scr_img.shape)>2):# 图像是RGB图像
        srcHeight, srcWidth, channels = scr_img.shape  # 获取原图的形状
        scr_img = np.pad(scr_img, ((0, 1), (0, 1), (0, 0)), 'constant')
        # 创建新图
        dst_img = np.zeros((dstHeight,dstWidth,channels),dtype = np.uint8)
        # 遍历新图,计算新图像素坐标在原图上的映射坐标
        for dstH in range(dstHeight):
            for dstW in range(dstWidth):
                scrH = (dstH)*(srcHeight/dstHeight)
                scrW = (dstW)*(srcWidth/dstWidth)
                # i,j = int(scrH),int(scrW)  # 获得映射到原图的位置的左上角的像素位置
                i,j = math.floor(scrH),math.floor(scrW)  # 获得映射到原图的位置的左上角的像素位置
                u,v = scrH-i,scrW-j  # 获得映射到原图的位置的到左上角的像素位置的距离 x距离和y距离
                dst_img[dstH][dstW] = (1-u)*(1-v)*scr_img[i][j] + (1-u)*v*scr_img[i][j+1] + u*(1-v)*scr_img[i+1][j] + u*v*scr_img[i+1][j+1]
        return  dst_img
    else:#图像是灰度图像
        srcHeight, srcWidth = scr_img.shape  # 获取原图的形状
        dst_img = np.zeros((dstHeight, dstWidth), dtype=np.uint8)
        # 遍历新图,计算新图像素坐标在原图上的映射坐标
        for dstH in range(dstHeight - 1):
            for dstW in range(dstWidth - 1):
                scrH = (dstH+0.5) * (srcHeight / dstHeight)-0.5 # +-0.5是为了保证映射中心对齐
                scrW = (dstW+0.5) * (srcWidth / dstWidth)-0.5
                i, j = int(scrH), int(scrW)  # 获得映射到原图的位置的左上角的像素位置
                u, v = scrH - i, scrW - j  # 获得映射到原图的位置的到左上角的像素位置的距离 x距离和y距离
                dst_img[dstH][dstW] = (1 - u) * (1 - v) * scr_img[i][j] + (1 - u) * v * scr_img[i][j + 1] + u * (
                            1 - v) * scr_img[i + 1][j] + u * v * scr_img[i + 1][j + 1]
        return dst_img


dstHeight,dstWidth = 500*2,500*2
dst_img = bilinear_interpolation(scr_img,dstHeight,dstWidth)
# dst_img = BiLinear_interpolation(scr_img,dstHeight,dstWidth)
cv2.imshow("dst_img",dst_img)
cv2.waitKey(0)