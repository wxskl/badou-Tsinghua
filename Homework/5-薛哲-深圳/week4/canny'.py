'''
canny边缘检测算法：
    Canny是目前最优秀的边缘检测算法， 其目标为找到一个最优的边缘， 其最优边缘的定义为：
        1、 好的检测： 算法能够尽可能的标出图像中的实际边缘
        2、 好的定位： 标识出的边缘要与实际图像中的边缘尽可能接近
        3、 最小响应： 图像中的边缘只能标记一次
    1. 对图像进行灰度化：
        方法1： Gray=(R+G+B)/3;
        方法2： Gray=0.299R+0.587G+0.114B;（这种参数考虑到了人眼的生理特点）
    2. 对图像进行高斯滤波：
        根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。 这样
        可以有效滤去理想图像中叠加的高频噪声。
    3. 检测图像中的水平、 垂直和对角边缘（如Prewitt， Sobel算子等） 。
    4 对梯度幅值进行非极大值抑制
    5 用双阈值算法检测和连接边缘
'''

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
thresh1表示最小阈值，thresh2表示最大阈值，用于进一步删选边缘信息
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 200, 300)

cv2.imwrite('canny.png', canny)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()
