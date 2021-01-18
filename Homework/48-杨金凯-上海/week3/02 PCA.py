import numpy as np

X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9,  35],
              [42, 45, 11],
              [9,  48, 5],
              [11, 21, 14],
              [8,  5,  15],
              [11, 12, 21],
              [21, 20, 25]])
class PCA:
    def __init__(self,X,K):
        self.X = X
        self.K = K
    def Centralized(self):
        '''
        样本中心化，样本值减去均值
        :return:
        '''
        # 求平均值
        mean = np.mean(self.X, axis=0)  # axis=0，计算每一列的均值
        # 中心化
        Center_X = self.X-mean
        print("中心化：\n",Center_X)
        return Center_X
    def Covariance(self):
        '''求中心化后数据的协方差'''
        Center_X = self.Centralized()
        m = Center_X.shape[0]
        Cov_Array = np.dot(Center_X.T,Center_X)*(1/(m-1))
        print("协方差矩阵：\n",Cov_Array)
        return Cov_Array
    def Eigenvalues_and_eigenvectors(self,X):
        '''求特征值与特征向量'''
        eigenvalue, eigenvectors = np.linalg.eig(X)
        print("特征值：\n",eigenvalue)
        print("特征向量：\n",eigenvectors)
        return eigenvalue,eigenvectors
    def Calculation_PCA(self):
        '''计算PCA'''

        # 获取协方差矩阵
        Cov_Array = self.Covariance()
        # 获取协方差矩阵的特征值和特征向量矩阵（列对应的是特特征向量）
        eigenvalue, eigenvectors = self.Eigenvalues_and_eigenvectors(Cov_Array)
        # 获取特征值从大到小排序下标
        # eigenvalue_sort = eigenvalue.argsort()[::-1]
        eigenvalue_sort = np.argsort(-1*eigenvalue)
        # 取前K个特征向量
        eigenvectors_Top_K = []
        for i in range(self.K):
            eigenvectors_Top_K.append(eigenvectors[:,eigenvalue_sort[i]].tolist())
        eigenvectors_Top_K = np.transpose(np.array(eigenvectors_Top_K))
        print("特征值排序：\n",eigenvalue_sort)
        print("特征值排序前K个特征值对应的特征向量：\n",eigenvectors_Top_K)
        # 用原始数据 点乘 前K个特征向量
        PCA_X = np.dot(self.X,eigenvectors_Top_K)
        print("PCA结果：\n",PCA_X)

pca = PCA(X,2)
# pca.Centralized()
# Cov_Array = pca.Covariance()
# pca.Eigenvalues_and_eigenvectors(Cov_Array)
pca.Calculation_PCA()