import numpy as np
import matplotlib.pyplot as plt


# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i)-min(x))/(max(x)-min(x)) for i in x]


def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


# 标准化
def z_score(x):
    '''
    x∗=(x−μ)/σ
    零均值归一化zero-mean normalization
    经过处理后的数据均值为0，标准差为1（正态分布）
    其中μ是样本的均值， σ是样本的标准差,标准差是总体各单位标准值与其平均数离差平方的算术平均数的平方根
    '''
    x_mean = np.mean(x)
    s2 = (sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x))**0.5
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
s3 = (sum([(i - np.mean(z)) * (i - np.mean(z)) for i in z]) / len(z))**0.5
print('标准差：',s3)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(n, cs)
plt.show()
