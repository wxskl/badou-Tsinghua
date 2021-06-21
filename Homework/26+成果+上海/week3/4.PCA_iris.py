# -*- coding: utf-8 -*-

'''
@author: chengguo
Theme：PCA算法之对鸢尾花数据降维处理并可视化
'''

#加载matplotlib用于数据的可视化
import matplotlib.pyplot as plt
#加载PCA算法包
import sklearn.decomposition as PCA
#加载鸢尾花数据
from sklearn.datasets import load_iris

x,y=load_iris(return_X_y=True)     #加载数据，x表示数据集中的属性数据，y表示数据标签
pca=PCA.PCA(n_components=2)            #加载pca算法，设置降维后的维度为2
reduced_x=pca.fit_transform(x)     #对原始数据进行降维，保存在reduced_x中
#三类数据点
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
#按鸢尾花的类别将降维后的数据点保存在不同的表中
for i in range(len(reduced_x)):
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
#对降维后的数据可视化
plt.scatter(red_x,red_y,c='r',marker='x')       #scatter()函数
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
