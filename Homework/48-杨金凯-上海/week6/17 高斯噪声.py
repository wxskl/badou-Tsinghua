import cv2
import random
def GaussianNoise(src,means,sigma,percetage):
    '''

    :param src: 要加噪声的图片
    :param means: 高斯噪声的均值
    :param sigma: 高斯噪声的标准差
    :param percetage: 需要加噪声的像素占总像素的比例
    :return: 加噪声后的图像
    '''
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])  # 计算需要加噪声的总像素数
    for i in range(NoiseNum):
        # 随机生成一个像素位置（randX，randY）
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)  # 随机生成行
        randY = random.randint(0, src.shape[1] - 1)  # 随机生成列
        # 此处在原有像素灰度值上加上随机高斯数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)  # 随机生成的像素位置加上高斯噪声
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


means = 10
sigma = 0.5
img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,means,sigma,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)


