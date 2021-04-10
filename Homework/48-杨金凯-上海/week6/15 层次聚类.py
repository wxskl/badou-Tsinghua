import numpy as np

class Hierarchical_Clustering:
    def __init__(self,data):
        self.data = data
        # print(data.shape)
        self.data_d = self.list_data_to_dictionary()  # 列表数据转换成字典数据
        self.data_num = len(data)          # 数据个数
        self.dim = len(data[0])      # 数据维度个数
        self.new_class_name = max(self.data_d.keys())
        self.Z = []
        print("数据个数：{0},数据维度：{1}".format(self.data_num,self.dim))

    def list_data_to_dictionary(self): # 列表转字典
        data_d = {}
        for i, value in enumerate(self.data):
            temp = np.array(value)
            if len(temp.shape) < 2:  # 如果列表维度是一维，则转换成二维
                data_d[i] = [value]
            else:
                data_d[i] = value
        return  data_d

    def calculation_euclidean_distance(self,coords1, coords2):
        """
        计算两个向量间的欧式距离 若是两个向量组则计算两个向量组间的平均距离
        :param coords1:[16.9]
        :param coords2:[[38.5],[39.5]]
        :return:
        """
        # print(coords1)
        # print(coords2)
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)

        sum_dis = 0
        for i in range(len(coords1)):
            for j in range(len(coords2)):
                # print((coords1[i], coords2[j]))
                dis = np.linalg.norm(np.array(coords1[i]) - np.array(coords2[j]), ord=2)
                # print(dis)
                sum_dis += dis
        avg_dis = sum_dis / (len(coords1) * len(coords2))
        # print(avg_dis)
        return avg_dis

    def european_matrix_and_get_min_value_index(self):
        """
        计算数据的欧式矩阵
        data:计算欧式矩阵的数据{k}
        return:
        """
        min_value_index = [0, 0]
        count = 0
        temp = np.float32("inf")
        for i in self.data_d.keys():
            for j in self.data_d.keys():
                count += 1
                if i == j:
                    # print(i, j)
                    # 用类来保存下标和数据间的距离。放在列表中
                    dis = np.float32("inf")
                    # dis_matrix.append(link_list_dis(i,j,dis))

                else:
                    # 计算两个数据组之间的距离
                    dis = self.calculation_euclidean_distance(self.data_d[i], self.data_d[j])
                    # dis_matrix.append( link_list_dis(i,j,dis))

                if dis < temp:     # 获得欧式矩阵的最小值及下标
                    temp = dis
                    # print("temp",temp)
                    min_value_index[0] = i      # 欧式矩阵的列
                    min_value_index[1] = j      # 欧式矩阵的行

        # print("距离",temp)
        # print("合并的类",min_value_index)
        # 根据要合并的下标调整生成新的字典数据
        # print("原字典数据",self.data_d)
        data_d_del1 = self.data_d.pop(min_value_index[0])
        data_d_del2 = self.data_d.pop(min_value_index[1])
        # print("删除合并项字典数据", self.data_d)
        new_class = [] # 用于保存本次的组合数据
        self.new_class_name += 1
        for i in range(len(data_d_del1)):  # 遍历组合数据data_d_del1
            new_class.append(data_d_del1[i])
        for i in range(len(data_d_del2)):  # 遍历组合数据data_d_del2
            new_class.append(data_d_del2[i])
        # print("合并项",new_class)
        # print("合并项后类名",self.new_class_name)
        self.data_d[self.new_class_name] = new_class
        # print("修正后的字典数据", self.data_d)
        self.Z.append([min_value_index[0],min_value_index[1],temp,len(new_class)])
        # print(self.Z)

    def HC(self):
        while len(self.data_d)>1:
            self.european_matrix_and_get_min_value_index()
        # print(self.Z)


# X = np.array([[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]])
# X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
X = [[0, 0], [0, 1], [1, 0],
     [0, 4], [0, 3], [1, 4],
     [4, 0], [3, 0], [4, 1],
     [4, 4], [3, 4], [4, 3]]
from sklearn import datasets
iris = datasets.load_iris()
# X = iris.data[:, :4]


HC = Hierarchical_Clustering(X)
HC.HC()

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
# f = fcluster(HC.Z,4,'distance')
# fig = plt.figure(figsize=(5, 3))
dn = dendrogram(HC.Z)
print(HC.Z)
plt.show()
