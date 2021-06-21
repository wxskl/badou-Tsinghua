# coding: utf-8
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs  #导入样本生成

"""
#初始化数据集 第一列球员每分钟助攻数，第二列是每分钟得分数
#列表不能用数组的方法读取
X = [[0.1888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]
X = np.array(X)   #数组可以用数组的方法读取
"""
#1千个样本，每个样本两个特征，3个族，指定了族中心和族方差
X,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1]],
                  cluster_std=[0.4, 0.3, 0.5], random_state=8)


clus = KMeans(n_clusters=3,n_init=10,algorithm='auto') #分成3类, 用不同的初始化质心运行算法的次数，默认是10.使用自动算法选择
y_read = clus.fit_predict(X) #使用配置好的参数对数据分类


print(clus)
print(y_read)
r_score = metrics.calinski_harabasz_score(X, y_read)   #评估聚类分数
print(r_score)

plt.scatter(X[:, 0], X[:, 1], c=y_read, marker='x')   #指定x和y,分类结果指定颜色,指定标记类型
plt.title("clus result")   #绘制标题
plt.ylabel("y") #y轴标签
plt.xlabel("x") #x轴标签
plt.show()   #显示图形


