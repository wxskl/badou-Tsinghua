import numpy as np
from sklearn.datasets import load_iris

'''
此算法很重要！！！！！要多次复习！！！！
特征提取的主要方法： PCA
特征选择（feature selection） 和特征提取（Feature extraction） 都属于降维（Dimension reduction)
传统的图像特征提取一般分为三个步骤： 预处理、 特征提取、 特征处理
特征处理： 主要目的是为了排除信息量小的特征， 减少计算量等： 常见的特征处理方法是降维，
常见的降维方法有：
1. 主成分分析（PCA） ；
2. 奇异值分解（SVD） ；
3. 线性判别分析
主成份分析算法PCA:
    就是将数据从原始的空间中转换到新的特征空间中， 例如原始的空间是三维的(x,y,z)， x、 y、 z
    分别是原始空间的三个基， 我们可以通过某种方法， 用新的坐标系(a,b,c)来表示原始的数据， 那么a、 b、 c
    就是新的基， 它们组成新的特征空间。 在新的特征空间中， 可能所有的数据在c上的投影都接近于0， 即可
    以忽略， 那么我们就可以直接用(a,b)来表示数据， 这样数据就从三维的(x,y,z)降到了二维的(a,b)。
    
    1. 对原始数据零均值化（中心化） ，中心化即是指变量减去它的均值， 使均值为0
    2. 求协方差矩阵，
    3. 对协方差矩阵求特征向量和特征值， 这些特征向量组成了新的特征空间。
    PCA算法的优化目标就是: ① 降维后同一维度的方差最大
                        ② 不同维度之间的相关性为0
    如果做了中心化， 则协方差矩阵为（中心化矩阵的协方差矩阵公式） ：D = (Z.I*Z)/m   Z为中心化后的矩阵，Z.I为Z的转置,m为单维度数据总量                    
'''

iris = load_iris()
data = iris.data


# print(data)  # 打印鸢尾花data数组


def zero_equalization(data):
    '''
    PCA步骤之零均值化（中心化）
    :param data: 输入数据
    :return: 零均值化后的数组
    '''
    # 求每一列的均值
    n = data.shape[0]
    m = data.shape[1]
    equal = np.array([0 for x in range(m)])  # 用于存放每一列的均值
    for x in range(m):
        temp = 0
        for y in range(n):
            temp = temp + data[y, x]  # 每一列累加和
            pass
        equal[x] = temp / n
        # print(equal[x])
        temp = 0
        pass
    for x in range(m):
        for y in range(n):
            data[y, x] = data[y, x] - equal[x]  # 每一列的元素减去这一列的均值,即零均值化
            pass
        pass

    return data


ZeroEqual = zero_equalization(data)


# print(ZeroEqual)

#  求协方差矩阵以及计算PCA------
def PCA(equaldata):
    '''

    :param equaldata:
    :return:
    '''
    m = data.shape[0]
    print(m)
    D = np.dot(equaldata.T, equaldata) / m

    # 求特征值，排序，k=2，求特征向量重组特征矩阵
    a, b = np.linalg.eig(D)  # a:特征值  b：特征向量
    # print(b)
    # 排序并得到排序前的下标
    a_index = np.argsort(-1 * a)
    # a_sort = np.sort(-1*a)
    # print(a)
    # print(a_index)

    # 原数组乘前k个特征向量得到降维的数据集
    b1 = b[a_index[0]]
    print(b1)
    b2 = b[a_index[1]]
    b_topK = np.transpose(np.vstack([b1, b2]))
    print(b_topK)
    PCA_data = np.dot(data, b_topK)
    return PCA_data


result = PCA(ZeroEqual)
print(result)