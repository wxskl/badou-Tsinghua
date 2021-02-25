import numpy as np
from sklearn.datasets import load_iris
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
        equal[x] = temp/n
        # print(equal[x])
        temp = 0
        pass
    for x in range(m):
        for y in range(n):
            data[y, x] = data[y, x] -equal[x]  # 每一列的元素减去这一列的均值,即零均值化
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
    D = np.dot(equaldata.T, equaldata)/m

    # 求特征值，排序，k=2，求特征向量重组特征矩阵
    a, b = np.linalg.eig(D)  # a:特征值  b：特征向量
    # print(b)
    # 排序并得到排序前的下标
    a_index = np.argsort(-1*a)
    # a_sort = np.sort(-1*a)
    # print(a)
    # print(a_index)

    # 原数组乘前k个特征向量得到降维的数据集
    b1 = b[a_index[0]]
    print(b1)
    b2 = b[a_index[1]]
    b_topK =np.transpose(np.vstack([b1, b2]))
    print(b_topK)
    PCA_data = np.dot(data, b_topK)
    return PCA_data

result = PCA(ZeroEqual)
print(result)