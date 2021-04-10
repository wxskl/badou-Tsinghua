import numpy as np
class KMeans:
    # 确定K值，即将数据集聚集成K个类簇或小组。
    def __init__(self,k,data):
        self.k = k
        self.data = data
        self.data_h = data.shape[0]
        self.data_w = data.shape[1]

    def calculation_euclidean_distance(self, coords1, coords2):
        """
        计算两个向量间的欧式距离
        :param x:
        :param y:
        :return:
        """
        return np.sqrt(np.sum((coords1 - coords2) ** 2))


    #  随机选择质心
    def random_choice_centroid(self):
        """
        随机选择质心
        :return:
        """
        index = []
        n = len(self.data)
        for i in range(self.k):
            temp = np.random.choice(range(n))
            while temp in index:  # 保证质心不同
                temp = np.random.choice(range(n))
            index.append(temp)
        return index
    def train_KMeans(self):
        '''
        :return: centroid,Class,labels
        '''
        # 初始化中心点
        index = self.random_choice_centroid()
        print(index)
        # index = [9, 11, 10]
        n = len(self.data)  # 数据长度

        centroid = []  # 保存中心点  质心
        for i in range(self.k):
            centroid.append(self.data[index[i]])
        centroid_prior = centroid

        while True:
        # for i in range(100):
            Class = [[] for _ in range(self.k)]  # 保存k个分类的数据
            labels = [-1 for _ in range(n)]
            for j in range(n):
                dis = []  # 存储临时距离
                for i in range(self.k):
                    dis.append(self.calculation_euclidean_distance(self.data[j], centroid[i])) # 计算一个数据到K个质心的距离
                Class_index = dis.index(min(dis))  # 求出 到最近一个质心的下标
                labels[j] = Class_index  # 记录数据self.data[j]所属类别
                Class[Class_index].append(np.array(self.data[j]))  # 找到距离最近的质心，并将数据添加到质心所对应的分类中

            # 从新计算质心
            dim = [[] for _ in range(self.k)]
            for i in range(self.k):
                for j in range(self.data_w):
                    # print([x[j] for x in Class[i]])获取第j列
                    # dim[i].append([x[j] for x in Class[i]])将获取第j列放入dim
                    dim[i].append(np.average(np.array([x[j] for x in Class[i]])))

            centroid = np.array(dim)
            if (centroid == centroid_prior).all():# 判断矩阵的被一个元素都相等 条件才成立
                break
            else:
                centroid_prior = centroid  # 保存质心
        return centroid,Class,labels# 返回质心 和 聚类列表


# 数据
X = np.array([[0.0888, 0.5885],
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
    ])
k = KMeans(3,X)  # 分类数设置为3
centroid,Class,labels = k.train_KMeans()
for i in range(len(centroid)):
    print(len(Class[i]))
print(Class)
print(labels)

import matplotlib.pyplot as plt
color=['red','b','g','y','c','m']
marker=['+','o','x','v','8','D']
# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
for i in range(len(Class)):
    x = [n[0] for n in Class[i]]
    y = [n[1] for n in Class[i]]


    ''' 
    绘制散点图 
    参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    '''
    plt.scatter(x, y, color =color[i] , marker = marker[i])

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()