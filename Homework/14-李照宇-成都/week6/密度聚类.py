import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:,:]  # #表示我们只取特征空间中的4个维度
#print(iris.data)

# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend('red')
plt.show()

#调用函数DBSCAN
dbs_can = DBSCAN(eps=0.4,min_samples=9)
dbs_can.fit(X)
data_pre = dbs_can.labels_
print(data_pre)
x0 = X[data_pre == 0]
x1 = X[data_pre == 1]
x2 = X[data_pre == 2]
x3 = X[data_pre == -1] #noise points

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:,0], x3[:,1], c="purple", marker='D', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()