import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
输入参数：cv2.kmeans(data，K， bestLabels，criteria，attempt，flags)
　　1. data：应该是np.float32类型的数据，每个特征应该放在一列。
　　2. K：聚类的最终数目
　　3. criteria：终止迭代的条件。当条件满足，算法的迭代终止。它应该是一个含有3个成员的元组，它们是（type，max_iter， epsilon）:
　　　　type终止的类型：有如下三种选择：
　　　　　　- cv2.TERM_CRITERIA_EPS 只有精确度epslion满足时停止迭代
　　　　　　- cv2.TERM_CRITERIA_MAX_ITER 当迭代次数超过阈值时停止迭代
　　　　　　– cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER 上面的任何一个条件满足时停止迭代
　　　max_iter：最大迭代次数
　　　epsilon：精确度阈值
　　4. attempts：使用不同的起始标记来执行算法的次数。算法会返回紧密度最好的标记。紧密度也会作为输出被返回
　　5. flags：用来设置如何选择起始中心。通常我们有两个选择：cv2.KMEANS_PP_CENTERS和 cv2.KMEANS_RANDOM_CENTERS。
    输出参数：
　　1. compactness：紧密度返回每个点到相应中心的距离的平方和
　　2. labels：标志数组，每个成员被标记为0，1等
　　3. centers：有聚类的中心组成的数组
'''
#读取原始图像灰度颜色
# img = cv2.imread('lenna.png', 0)
# cv2.imshow('lenna.png',img)
# cv2.waitKey(0)
# rows,cols = img.shape[:]
# #transfer to 1 dimension
# data = img.reshape(rows*cols,1)
# data = np.float32(data)


#读取BGR色
img = cv2.imread('lenna.png')
data = img.reshape((-1,3))
data = np.float32(data)


#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#设置起始中心
flags = cv2.KMEANS_RANDOM_CENTERS
#Kmeans聚类
compactness,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)

#单通道dst
#dst = labels.reshape((img.shape[0],img.shape[1]))

#三通道dst
centers = np.uint8(centers) #去小数
res = centers[labels]#(512*512,3)，根据label，一位一位取特征值到res里
dst = res.reshape((512,512,3))
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#显示图像
titles = [u'原始图像','聚类图象']
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i]),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

