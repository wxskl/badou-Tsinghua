###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
(1) 将每个对象看作一类， 计算两两之间的最小距离；
(2) 将距离最小的两个类合并成一个新类；
(3) 重新计算新类与所有类之间的距离；
(4) 重复(2)、 (3)， 直到所有类最后合并成一类。
'''

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)，这个方法的使用在层次聚类之前。当s和t行程一个新的聚类簇u时，
s和t被从森林（已经形成的聚类簇群）中移除，而用新的聚类簇u来代替。当森林中只有一个聚类簇时算法停止。而这个聚类簇就成了聚类树的根。
返回值为数组共有四列组成，第一字段与第二字段分别为聚类簇的编号，在初始距离前每个初始值被从0~n-1进行标识，
每生成一个新的聚类簇就在此基础上增加一对新的聚类簇进行标识，第三个字段表示前两个聚类簇之间的距离，第四个字段表示新生成聚类簇所包含的元素的个数。

1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法：
    (1)single:最近邻,把类与类间距离最近的作为类间距
    (2)complete:最远邻,把类与类间距离最远的作为类间距
    (3)average:平均距离,类与类间所有pairs距离的平均
    (4)ward:最小方差算法

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
此函数用来将聚类层次划分好的矩阵即Z具体分成几个类，类之间的距离阈值是t
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
print(f)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()