import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat.png")

data = img.reshape((-1,3))  #-1表示不知道多少行，3表示排成3列,3个属性要聚类

data = np.float32(data)    #转换成float32类型，小数类型

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #确定迭代的终止条件,最多迭代10次，精度为1
K = 6 #分成的类
attempts = 10 #重复10次算法，产生最佳结果
flags = cv2.KMEANS_RANDOM_CENTERS #随机选择起始中心
bestLabels = np.mat([])
compactness,labels,centers = cv2.kmeans(data,K,bestLabels,criteria,attempts,flags) #返回紧密度，分类标签，分成的每个类的中心
centers = np.uint8(centers)
labels = labels.flatten()  #没折叠之前是一个1列的矩阵，二维，折叠之后是一个一维的数据
res = centers[labels] #每一列的值是以labels作为索引，取出centers中的值，填到对应行的位置上，3列RGB做同样的操作
res = res.reshape(img.shape) #还原成图像的形状

cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
