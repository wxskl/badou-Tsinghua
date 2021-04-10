import cv2
from numpy import shape
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		#每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]<0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,5,4,1)
#img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)


# import cv2
# import numpy as np
# import random
#
# def GaussionNoise(src,means,sigma,percentage):
#     Opb = src[:,:,0]
#     Opg = src[:,:,1]
#     Opr = src[:,:,2]
#     NoiseNum = int(percentage*src.shape[0]*src.shape[1])
#     for i in range(NoiseNum):
#         radx = random.randint(0,src.shape[0]-1)
#         rady = random.randint(0,src.shape[1]-1)
#         Opb[radx,rady] = Opb[radx,rady]+random.gauss(means,sigma)
#         Opg[radx,rady] = Opg[radx, rady] + random.gauss(means, sigma)
#         Opr[radx,rady] = Opr[radx, rady] + random.gauss(means, sigma)
#         if Opb[radx,rady]<0:
#             Opb[radx,rady]=0
#         elif Opb[radx,rady]>255:
#             Opb[radx,rady]=255
#         if Opg[radx,rady]<0:
#             Opg[radx,rady]=0
#         elif Opg[radx,rady]>255:
#             Opg[radx,rady]=255
#         if Opr[radx,rady]<0:
#             Opr[radx,rady]=0
#         elif Opr[radx,rady]>255:
#             Opr[radx,rady]=255
#     Noisesig = cv2.merge([Opb,Opg,Opr])
#     return Noisesig
# img = cv2.imread('lenna.png',1)
# img2 = GaussionNoise(img,0,4,0.5)
# #img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# #img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow('source',img2)
# #cv2.imshow('lenna_GaussianNoise',img3)
# cv2.waitKey(0)
