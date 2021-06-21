# -*- encoding=UTF-8 -*-
import numpy as np

class NeuralNetwork():
    def __init__(self,input_node_num,hidden_node_num,output_node_num,learning_rate):
        self.inode_num = input_node_num
        self.hnode_num = hidden_node_num
        self.onode_num = output_node_num
        #self.activation_function = lambda x:activateMethod.relu(x)
        self.lr = learning_rate
        self.activation_function = lambda x: activateMethod.sigmoid(x)
        self.wih = np.array([[0.15,0.20],[0.25,0.30]])
        self.who = np.array([[0.40,0.45],[0.50,0.55]])
        self.bias = np.array([[0.35],[0.60]])

    def train(self,input_nodes,input_labels):
        input_nodes = np.array(input_nodes,ndmin=2).T
        input_labels = np.array(input_labels,ndmin=2).T
        #推理
        hnodes = self.activation_function(np.dot(self.wih,input_nodes) + self.bias[0])
        onodes = self.activation_function(np.dot(self.who,hnodes) + self.bias[1])

        #训练
        # 计算一下误差
        output_errors = input_labels - onodes
        self.who = self.who + self.lr * np.dot(output_errors * onodes * (1 - onodes), hnodes.T)
        self.bias[1] = self.bias[1] + self.lr * np.dot(output_errors.T, onodes * (1 - onodes))

        hidden_errors = np.dot(self.who, output_errors * onodes * (1 - onodes))
        self.wih = self.wih + self.lr * np.dot(hidden_errors * hnodes * (1 - hnodes), input_nodes.T)
        self.bias[0] = self.bias[0] + self.lr * np.dot(hidden_errors.T , hnodes * (1 - hnodes))

    def query(self,test_images):
        #先写个推理的代码吧
        self.hnodes = self.activation_function(np.dot(self.wih,test_images))
        self.onodes = self.activation_function(np.dot(self.who,self.hnodes))
        return self.onodes

    # def categorical(self):
    #     print("activateMethod.softmax(self.onodes)",activateMethod.softmax(self.onodes))

import scipy.special as S
class activateMethod():
    @staticmethod
    def sigmoid(x):
        return S.expit(x)

    @staticmethod
    def relu(x):
        return max(0,x)

    @staticmethod
    def softmax(x):
        return S.softmax(x)


if __name__ == "__main__":
    input_num = 2
    hidden_num = 2
    output_num = 2
    learning_rate = 0.5
    epoch = 15
    input_nodes = [0.05,0.10]
    input_labels = [0.01,0.99]

    network = NeuralNetwork(input_num,hidden_num,output_num,learning_rate)
    for i in range(epoch):
        network.train(input_nodes,input_labels)

    test_images = [[0.06],[0.12]]
    print("network.query(test_images)",network.query(test_images))
    print("network.wih",network.wih)
    print("network.who",network.who)
    print("network.bias",network.bias)

    # network.categorical()