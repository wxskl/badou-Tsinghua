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
        :param coords1:向量1
        :param coords2:向量2
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

        # while True:
        for i in range(20):
            print(i)
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
                    dim[i].append(np.average(np.array([x[j] for x in Class[i]])))
            centroid = np.array(dim)
            if (centroid == centroid_prior).all():  # 判断矩阵的被一个元素都相等 条件才成立
                break
            else:
                centroid_prior = centroid  # 保存质心
        return centroid,Class,labels  # 返回质心 和 聚类列表
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # 读取原始图像灰度颜色
    img = cv2.imread('lenna.png', 0)
    print(img.shape)

    # 获取图像高度、宽度
    rows, cols = img.shape[:]

    # 图像二维像素转换为一维
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)

    k = KMeans(4, data)
    centroid, Class, labels = k.train_KMeans()
    # 生成最终图像
    labels = np.array((labels))
    dst = labels.reshape((img.shape[0], img.shape[1]))
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    images = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
