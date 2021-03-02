import cv2
import numpy as np
from PIL import  Image
from skimage import util

'''
random_noise()
mode:表示要添加的噪声类型
    gaussian：高斯噪声
    localvar:高斯分布的加性噪声，在图像的每个点处具有指定的局部方差
    poisson：泊松噪声
    salt：盐噪声，随机将像素值变为1
    pepper：椒噪声，随机将像素值变为0或-1，取决于矩阵的值是否带符号
    s&p:椒盐噪声
    speckle：均匀噪声
seed：可选的，int型，在生成噪声前会先设置随机种子
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
alt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
'''

img = cv2.imread("cat.png")
noise_img = util.random_noise(img, mode='s&p') #加入椒盐噪声
cv2.imshow("s&p", noise_img)
cv2.imshow("source",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
