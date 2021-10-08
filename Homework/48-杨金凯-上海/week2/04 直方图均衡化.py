import cv2
import matplotlib.pyplot as plt

def histogram(img_gray):
    '''
    求灰度图像的直方图
    :param img_gray: 灰度图像
    :return:
    '''
    histogram = [0 for x in range(256)]
    h,w = img_gray.shape
    for i in range(h):
        for j in range(w):
            histogram[img_gray[i][j]] = histogram[img_gray[i][j]]+1
    return histogram

def histogram_equalization(img_gray):
    '''
    灰度图直方图均衡化
    :param img_gray: 灰度图像
    :return: 灰度图直方图均衡化后的图像
    '''
    histogram_data = histogram(img_gray)
    Pi = [0 for x in range(256)]
    SumPi = []
    sum = 0
    h, w = img_gray.shape
    total_pixels = h*w
    for i in range(len(histogram_data)):
        Pi[i] = histogram_data[i] / total_pixels
        sum = sum + Pi[i]
        SumPi.append(sum)
    equalization = {}
    for i in range(len(histogram_data)):
        equalization[i] = round(SumPi[i] * 256-1 )
    # return equalization
    for i in range(h):
        for j in range(w):
            img_gray[i][j] = equalization[img_gray[i][j]]
    return img_gray


# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

img = cv2.imread('lenna.png',1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(221)
plt.imshow(img_gray,cmap = "gray")
plt.title("灰度图")

plt.subplot(222)
hist_x = [x for x in range(256)]
plt.xticks(range(0,255,25))
hist = histogram(img_gray)
plt.bar(hist_x,hist,width=1)
plt.title("灰度直方图")


plt.subplot(223)
img_equal = histogram_equalization(img_gray)
plt.imshow(img_equal,cmap = "gray")
plt.title("均衡化")

plt.subplot(224)
hist_equal = histogram(img_equal)
plt.bar(hist_x,hist_equal,width=1)
plt.title("均衡化后的直方图")
plt.show()

def RGB_histogram(img):
    '''
    求彩色图像的直方图
    :param img: BGR图像或者RGB图像
    :return:
    '''
    Bhistogram = histogram(img[:,:,0])
    Ghistogram = histogram(img[:,:,1])
    Rhistogram = histogram(img[:,:,2])
    return Bhistogram,Ghistogram,Rhistogram

def RGB_histogram_equalization(img):
    img[:, :, 0] = histogram_equalization(img[:,:,0])
    img[:, :, 1] = histogram_equalization(img[:,:,1])
    img[:, :, 2] = histogram_equalization(img[:,:,2])
    return img
plt.subplot(221)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.title("彩色图")

plt.subplot(222)
BGRhistogram = RGB_histogram(img)
plt.bar(hist_x,BGRhistogram[0],color="blue",width=1)
plt.bar(hist_x,BGRhistogram[1],color="green",width=1)
plt.bar(hist_x,BGRhistogram[2],color="red",width=1)
plt.title("彩色图的直方图")

plt.subplot(223)
RGB_img_equal = RGB_histogram_equalization(img)
RGB_img_equal1 = cv2.cvtColor(RGB_img_equal,cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img_equal1)
plt.title("彩色图均衡化")

plt.subplot(224)
RGB_hist_equal = RGB_histogram(RGB_img_equal)
plt.bar(hist_x,RGB_hist_equal[0],color="blue",width=1)
plt.bar(hist_x,RGB_hist_equal[1],color="green",width=1)
plt.bar(hist_x,RGB_hist_equal[2],color="red",width=1)
plt.title("彩色图均衡化后的直方图")
plt.show()
cv2.imshow("RGB_img_equal",RGB_img_equal)
cv2.waitKey(0)
