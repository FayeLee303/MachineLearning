"""学习向量量化，数据样本有类标签"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import random
import math
from collections import Counter

from sklearn.datasets import load_iris

# 读取数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:,:])
    return data[:,:-1], data[:,-1] # X Y

"""
输入 样本集合X，X有m个样本点，每个样本点是n维，标签集合Y是m*1
    原型向量个数k（分成k个簇），各个原型向量预设的类别标记{t1...tk}
    学习率[0,1]，迭代次数
输出 一组原型向量{p1...pk}，pi是个n维向量，其实就是每个簇的中心点
过程
    初始化一组原型向量{p1...pk}
    重复
        随机选择样本点xj,yj
        计算xj和pi的距离，欧氏距离
        找出离xj最近的原型向量pi
        如果yj=ti
            new_p = pi + learningrate*(xj-pi)
        else
            new_p = pi - learningrate*(xj-pi)
        更新pi = new_p
    直到满足停止条件
"""
class LVQ(object):
    def __init__(self,learning_rate=0.1):
        self.model = None
        self.learning_rate = learning_rate

    # 距离度量
    def distance(self, x1, x2):
        return math.sqrt(sum(pow(x1 - x2, 2)))

    # 初始化原型向量
    def init_vectors(self,k,n):
        vectors = np.zeros((k,n))
        return vectors

    # 随机选择样本点，返回index就行了
    def random_index(self,m):
        index = random.randint(0,m-1)
        return index

    # 训练
    def train(self,X,Y,k,set_labels,max_iter):
        """
        :param X: 样本点，m行，n维
        :param Y: 样本真实标签，m行，1列
        :param k: 要分成k个簇
        :param set_labels: 预设k个簇的标签
        :return: 一组原型向量
        """
        m, n = X.shape  # m个样本,n维

        # 初始化原型向量
        vectors = self.init_vectors(k,n)

        for temp in range(max_iter):
            # 随机选择样本点
            index = self.random_index(m)
            xj,yj = X[index],Y[index]
            # 计算样本点xj和所有中心点的距离
            distances = []
            for i in range(k):
                distances.append(self.distance(xj, vectors[i]))  # 计算距离
            nearest_distance = np.min(distances)  # 最近的距离
            nearest_index = distances.index(nearest_distance)  # 下标
            nearest_vector = vectors[nearest_index]  # 最近的向量
            set_label = set_labels[nearest_index]   # 最近的向量对应的预设标签
            new_vector = None
            # 学习
            if set_label == yj:
                new_vector = nearest_vector + self.learning_rate * (xj - nearest_vector)
            else:
                new_vector = nearest_vector - self.learning_rate * (xj - nearest_vector)
            vectors[nearest_index] = new_vector # 更新向量

        return vectors

    # 根据得到的原型向量进行分类
    def clustering(self,X,Y,k,set_labels,max_iter):
        vectors = self.train(X,Y,k,set_labels,max_iter)

        # 初始化簇集合
        clusters = {}
        for i in range(k):
            clusters[i] = []
        m,n = X.shape

        # 分簇
        for j in range(m):
            distances = []
            # 计算样本点xj和所有中心点的距离
            for i in range(k):
                distances.append(self.distance(X[j],vectors[i]))   # 计算距离
            nearest_distance = np.min(distances)  # 最近的距离
            nearest_index = distances.index(nearest_distance)  # 是第几个

            clusters[nearest_index].append(X[j])  # 分簇

        # 显示
        self.show(clusters)

    # 显示分簇的结果
    def show(self,clusters):
        k = len(clusters)
        colors = ['red','green','blue','yellow','orange','pink','purple']
        for i in range(k):
            ci = np.array(clusters[i])
            plt.scatter(ci[:, 0], ci[:, 1], c=colors[i], label='cluster_%d'%i)

        plt.title('clustering')
        plt.legend()
        # plt.show()

if __name__ == "__main__":
    X,labels = create_data()
    model = LVQ(learning_rate=0.1)
    model.clustering(X,labels,3,[0,1,2],200)
    plt.show()