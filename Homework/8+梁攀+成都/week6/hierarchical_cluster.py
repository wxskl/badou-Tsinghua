from scipy.cluster.hierarchy import dendrogram,linkage,fcluster,distance
from matplotlib import pyplot as plt
import scipy
import numpy as np

points = np.random.randn(10,4)   #生成20个样本，每个样本4维
dist_mat = distance.pdist(points, 'euclidean')  #生成点与点之间的距离矩阵，这里使用欧式距离
print(dist_mat)

Z = linkage(dist_mat, method='ward') #进行层次聚类, 使用ward方法,返回值是聚类树的合并过程
print(Z)

f = fcluster(Z,2,criterion='distance') #根据阈值决定分类，不同的阈值会导致不同类的合并，判断标准是距离，返回值是每个点的类型标记
print(f)

den = dendrogram(Z)   #将层次聚类结果以树状图表示出来
plt.show()