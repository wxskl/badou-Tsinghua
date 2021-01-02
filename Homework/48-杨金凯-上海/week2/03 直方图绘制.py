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


# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
###读取灰度图并显示
# img_gray = cv2.imread('lenna.png',0)
img = cv2.imread('lenna.png',1)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(221)
plt.imshow(img_gray,cmap = "gray")
### 计算灰度直方图并显示
plt.subplot(222)
histogram_x = [x for x in range(256)]
Gray_histogram = histogram(img_gray)
plt.bar(histogram_x,Gray_histogram,width=2)
# 显示横轴标签
plt.xlabel("像素值")
# 显示纵轴标签
plt.ylabel("像素个数")
# 显示图标题
plt.title("灰度直方图")


# 计算彩色直方图并显示
BGRhistogram = RGB_histogram(img)
plt.subplot(223)
plt.bar(histogram_x,BGRhistogram[0],color="blue",width=2)
plt.bar(histogram_x,BGRhistogram[1],color="green",width=2)
plt.bar(histogram_x,BGRhistogram[2],color="red",width=2)

## API标准代码1
#-----------------------------------------------------
plt.subplot(224)
img = img_gray.reshape(-1)  #将图像展开成一个一维的numpy数组
plt.hist(img, 256,width=2)  #将数据分为256组
plt.show()
## API标准代码2
#-----------------------------------------------------
hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()


## API标准代码3
#-----------------------------------------------------
image = cv2.imread("lenna.png")
cv2.imshow("Original",image)
#cv2.waitKey(0)

chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()