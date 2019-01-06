"""学习向量量化，数据样本有类标签"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import math

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
    def __init__(self,learning_rate=0.01):
        self.model = None
        self.learning_rate = learning_rate

    # 距离度量
    def distance(self, x1, x2):
        return math.sqrt(sum(pow(x1 - x2, 2)))

    # 初始化原型向量
    def init_vectors(self,k,n):
        # vectors = np.zeros((k,n))
        # vectors = np.random.rand(k,n)+3
        # vectors = np.random.randint(0,6,(k, n))
        vectors = np.ones((k,n))*3
        # vectors = np.random.random_sample((k,n))*3
        print('init vectors',vectors)
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
                # print('+')
                new_vector = nearest_vector + self.learning_rate * (xj - nearest_vector)
            else:
                # print('-')
                new_vector = nearest_vector - self.learning_rate * (xj - nearest_vector)
            vectors[nearest_index] = new_vector # 更新向量

            # if temp in [1,2,3,4]:
            #     print('xj',xj,'yj',yj)
            #     print('nearest_vector',nearest_vector,'set_label',set_label)
            #     print('new vector', new_vector)
            #     print('vectors', vectors)
        print('vectors',vectors)
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

        # print('clusters',clusters)

        # 显示
        self.show_vectors(vectors)
        self.show_result(clusters)

        # 指标
        print('DBI', self.calculate_DBI(k, clusters, vectors))
        print('DI', self.calculate_DI(k, clusters))

    # 显示原型向量，就是中心点
    def show_vectors(self,vectors):
        colors = ['red', 'green', 'blue', 'yellow','pink', 'orange',  'purple']
        for i in range(vectors.shape[0]):
            plt.scatter(vectors[i, 0], vectors[i, 1], c=colors[i], marker='*')

    # 显示分簇的结果
    def show_result(self,clusters):
        k = len(clusters)
        colors = ['red','green','blue','yellow','pink','orange','purple']
        for i in range(k):
            ci = np.array(clusters[i])
            print(ci.shape)
            if ci != []:
                plt.scatter(ci[:, 0], ci[:, 1], c=colors[i], label='cluster%d'%i)

        plt.title('clustering')
        plt.legend()
        # plt.show()

    # 计算簇间样本的平均距离
    def calculate_avg(self, cluster):
        X = np.array(cluster)
        m, n = X.shape  # m是该簇里的样本个数
        dist = np.zeros((m, m))  # 距离矩阵
        for i in range(m):
            for j in range(m):
                dist[i, j] = self.distance(X[i], X[j])
        sum = np.sum(dist)
        return 2 / (m * (m - 1)) * sum

    # 计算簇内样本的最远距离
    def calculate_diam(self, cluster):
        X = np.array(cluster)
        m, n = X.shape
        dist = np.zeros((m, m))  # 距离矩阵
        for i in range(m):
            for j in range(m):
                dist[i, j] = self.distance(X[i], X[j])
        return np.max(dist)

    # 计算簇Ci和簇Cj的最近样本（边缘）的距离
    def calculate_dmin(self, cluster_1, cluster_2):
        X1 = np.array(cluster_1)
        X2 = np.array(cluster_2)
        # m1是簇1的样本个数，m2是簇2的样本个数
        m1, m2 = X1.shape[0], X2.shape[0]
        dist = np.zeros((m1, m2))  # 距离矩阵
        for i in range(m1):
            for j in range(m2):
                dist[i, j] = self.distance(X1[i], X2[j])
        return np.min(dist)

    # 计算簇Ci和簇Cj的中心点p1p2的距离
    def calculate_dcen(self, i, j, P):
        return self.distance(P[i], P[j])

    # 计算DBI
    def calculate_DBI(self, k, clusters, P):
        temp1 = np.zeros((1, k))
        for i in range(k):
            if clusters[i]==[]:
                continue
            temp2 = np.zeros((1, k))
            for j in range(k):
                if j == i or clusters[j]==[]:  # 算簇间中心点的距离dcen所以i=j时跳过
                    continue
                else:
                    temp2[:, j] = (self.calculate_avg(clusters[i]) +
                                   self.calculate_avg(clusters[j])) / \
                                  self.calculate_dcen(i, j, P)
            temp1[:, i] = np.max(temp2)
        sum = np.sum(temp1)
        return 1 / k * sum

    # 计算DI
    def calculate_DI(self, k, clusters):
        temp1 = np.ones((1, k))
        for i in range(k):
            if clusters[i] == []:
                continue
            # temp2 = np.zeros((1, k))
            temp2 = np.ones((1, k))  # 因为下面要求最小，如果初始化是0，就会都变成0
            for j in range(k):
                temp3 = np.ones((1, k))
                for l in range(k):
                    if clusters[l] == []:
                        continue
                    temp3[:, l] = self.calculate_diam(clusters[l])
                if j == i or clusters[j] == []:  # 算簇间dmin所以i=j时跳过
                    continue
                else:
                    temp2[:, j] = self.calculate_dmin(clusters[i], clusters[j]) / np.max(temp3)
            temp1[:, i] = np.min(temp2)
        return np.min(temp1)

if __name__ == "__main__":
    X,labels = create_data()
    inputs_sepal = X[:, :2]
    inputs_petal = X[:, 2:4]

    model = LVQ(learning_rate=0.005)

    model.clustering(X,labels,3,[0.0,1.0,2.0],500)
    # plt.show()