import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X)
# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  
'''

'''
需要两个参数： ε (eps) 和形成高密度区域所需要的最少点数 (minPts)
• 它由一个任意未被访问的点开始， 然后探索这个点的 ε-邻域， 如果 ε-邻域里有足够的点， 则建立一
个新的聚类， 否则这个点被标签为杂音。
• 注意， 这个点之后可能被发现在其它点的 ε-邻域里， 而该 ε-邻域可能有足够的点， 届时这个点会被
加入该聚类中。
'''
dbscan = DBSCAN(eps=0.4, min_samples=9)  # 聚类函数类实例化

dbscan.fit(X)  # 将所有数据分类并返回分类的标签，噪声为-1
label_pred = dbscan.labels_
print(label_pred)
print(label_pred == 0)
# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == -1]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='+', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()