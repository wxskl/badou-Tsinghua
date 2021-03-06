from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数:
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None)
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息;
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
    #method是指计算类间距离的方法,比较常用的有3种:
    #single:最近邻,把类与类间距离最近的作为类间距
    #average:平均距离,类与类间所有pairs距离的平均
    #complete:最远邻,把类与类间距离最远的作为类间距
f = fcluster(Z,4,'distance')
#fig = plt.figure(figsize=(5, 3)) #调整输出图大小
dn = dendrogram(Z)
print(Z)
plt.show()
print(f)


# import scipy
# import scipy.cluster.hierarchy as sch
# from scipy.cluster.vq import vq,kmeans,whiten
# import numpy as np
# import matplotlib.pylab as plt
#
# #生成待聚类的数据点,这里生成了20个点,每个点4维:
# points=np.random.randn(20,4)
# #层次聚类
# #生成点与点之间的距离矩阵,这里用的欧氏距离:
# disMat = sch.distance.pdist(points,'euclidean')
# #进行层次聚类:
# Z=sch.linkage(disMat,method='average')
# #将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
# P=sch.dendrogram(Z)
# plt.savefig('plot_dendrogram.png')
# #根据linkage matrix Z得到聚类结果:
# cluster= sch.fcluster(Z, t=4,criterion='distance')

#print("Original cluster by hierarchy clustering:\n",cluster)