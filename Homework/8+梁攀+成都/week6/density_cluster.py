import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4] #取鸢尾花的四个特征
X[:,:4] = (X[:,:4] - X[:, :4].min())/(X[:, :4].max() - X[:, :4].min())   #数据归一化

dbscan = DBSCAN(eps=0.07, min_samples=4)  #指定领域距离，领域内的最少点数
cluster = dbscan.fit(X)  #进行密度聚类
label_pred = cluster.labels_ #返回每一个样本的分类结果
print(label_pred)   #-1表示离群点


x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == -1]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="black", marker='x', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

