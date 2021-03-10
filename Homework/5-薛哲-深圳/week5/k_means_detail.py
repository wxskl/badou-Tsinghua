'''
k-means聚类算法
第一步， 确定K值， 即将数据集聚集成K个类簇或小组。
第二步， 从数据集中随机选择K个数据点作为质心（Centroid） 或数据中心。
第三步， 分别计算每个点到每个质心之间的距离， 并将每个点划分到离最近质心的小组。
第四步， 当每个质心都聚集了一些点后， 重新定义算法选出新的质心。 （对于每个簇， 计
算其均值， 即得到新的k个质心点）
第五步， 迭代执行第三步到第四步， 直到迭代终止条件满足为止（分类结果不再变化）
'''

import numpy as np
import random
#  准备数据，从数据集中随机选择K个数据点作为质心（Centroid） 或数据中心。
data = [[0, 0], [1, 2], [3, 1], [8, 8], [9, 10], [10, 7]]
# 确定k值
k = 2
centers = np.array(random.sample(data, k))  # 随机取k个作为中心
print('centers: ', centers)


def my_k_means(data, k):
    dst = {}

    for i in range(len(data)):

        temp = np.array([])
        for j in range(k):
            temp = np.append(temp, ((data[i][0]-centers[j, 0])**2 + (data[i][1]-centers[j, 1])**2)**0.5)
        # print(temp)
        index = np.argmin(temp)
        # print(i)
        # print(index)

        # print(dst1)
        if index not in dst.keys():
            dst[index] = np.array(data[i]).reshape(1, 2)
        else:
            dst[index] = np.vstack((dst[index], np.array([data[i][0], data[i][1]])))
        # print(dst)
    return dst
# out = my_k_means(data, k)
# 计算每个簇的均值作为新的中心,重新聚类直到聚类结果不再变化
flag = True
while(flag):
    out = my_k_means(data, k)
    print(out)
    for i in range(k):
        classes = np.array(out[i])  # 拿到每一类的聚类结果
        # print(classes)
        x, y = np.mean(classes[:, 0]), np.mean(classes[:, 1])  # 默认数据只有两个维度的特征值
        # print(x,y)
        centers[i] = [x, y]
    print('new_center: ', centers)
    out_after = my_k_means(data, k)
    for i in range(k):
        if out_after[i].all() != out[i].all():
            break
        elif i == k-1 and out_after[i].all() == out[i].all():
            flag = False

print('out: ', out)