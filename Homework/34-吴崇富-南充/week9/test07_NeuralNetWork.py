#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        # self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        # self.who = np.random.rand(self.onodes,self.hnodes)-0.5
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self,inputs_list,targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih,inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who,hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        output_errors1 = (targets - final_outputs)*final_outputs*(1 - final_outputs)
        hidden_errors = np.dot(self.who.T,output_errors1)
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr*np.dot(output_errors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))
    def query(self,inputs):
        # 根据输入数据计算并输出答案
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
with open('dataset/mnist_train.csv','r') as f:
    training_data_list = f.readlines()

#加入epocs,设定网络的训练循环次数
epochs = 10
for e in range(epochs):
    # 把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:])/255.0*0.99+0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

with open('dataset/mnist_test.csv','r') as f:
    test_data_list = f.readlines()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    #预处理数字图片
    inputs = np.asfarray(all_values[1:])/255.0*0.99+0.01 # 注意是np.asfarray返回转换为浮点类型的数组
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print('performance = ',scores_array.sum()/scores_array.size)

'''
np.array与np.asarray都可以将list数据转化为ndarray。
当参数为list数据时，list数据转化为array时，其两者的结果是一致的；
当参数为ndarray时，np.array会复制一个ndarray对象，np.asarray不会复制，而是和原来的占用同一个内存
'''
