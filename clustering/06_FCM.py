"""模糊聚类算法"""
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
输入 数据集D m个样本，n维，无监督，不需要标签
    聚类簇数k, 迭代次数b=0,精度e
输出 软划分矩阵U
    一组原型向量{p1...pk}，pi是个n维向量，其实就是每个簇的中心点
过程
    随机初始化软划分矩阵U，每个元素在0到1之间，每一列之和等于1
    初始化一组聚类原型向量
    重复
        计算划分矩阵U
        计算原型向量P
        如果|pb -pb+1| < e
            停止，并输出U和P
        否则
            b+=1   
"""
class FCM(object):
    def __init__(self,q=2):
        self.q = q  # 加权指数，平滑参数，默认为2

    # 距离度量
    def distance(self, x1, x2):
        return math.sqrt(sum(pow(x1 - x2, 2)))

    # 初始化划分矩阵
    # 每个元素是随机的，uij取[0,1],每一列之和等于1
    def init_divide_matrix(self,k,m):
        # 假设随机生成5个数，每个数在0到1之间，5个数之和为1
        # a = np.zeros((1, 5))
        # print(a)
        # a[0][0] = np.random.uniform(0, 1)
        # print(a)
        # for i in range(1, 4):
        #     temp = sum(a[0, :i])
        #     a[0][i] = np.random.uniform(0, 1 - temp)
        #     print(a)
        # print(a)
        # temp = sum(a[0, :4])
        # a[0][4] = 1 - temp
        # print(a)
        # print(sum(a[0, :]))
        U = np.zeros((k, m))
        # 每列第一个随机(第一行是完全随机的)，每列最后一个等于1-前面的
        # 中间的随机，随机范围0到1-前面的
        U[0,:] = np.random.uniform(0, 1,(1,m))  # 第一行是完全随机的
        for j in range(m):
            for i in range(1,k-1):
                temp = sum(U[:i, j])
                U[i][j] = np.random.uniform(0, 1-temp)
        temp = np.sum(U[:k-1,:],axis=0)
        U[k-1,:] = np.ones((1,m)) - temp    # 最后一行等于1-前面的
        # print(U)
        # print(np.sum(U[:,:],axis=0))
        return U

    # 初始化原型向量
    def init_vectors(self,k,n):
        vectors = np.zeros((k, n))
        return vectors

    # 初始化簇集合
    def init_clusters(self,k):
        clusters = {}
        for i in range(k):
            clusters[i] = []
        return clusters

    # 训练
    def train(self,X,k,max_iter=200):
        """
        :param X: 样本集，m个样本，n维
        :param k: 分簇个数
        :param e: 精度
        :return:
        """
        m,n = X.shape   # m个样本，n维
        U = self.init_divide_matrix(k,m)  # 划分矩阵
        P = self.init_vectors(k,n)  # 原型向量
        b = 0   # 迭代次数

        # print('初始U',U)
        # print('初始P',P)

        for iter in range(max_iter):

            # 计算原型向量P
            for i in range(k):
                up, down = 0, 0  # 分子分母
                for j in range(m):
                    up += pow(U[i][j], self.q) * X[j]
                    down += pow(U[i][j], self.q)
                P[i] = up / down

            # 计算划分矩阵U
            for i in range(k):
                for j in range(m):
                    dij = self.distance(X[j],P[i])
                    temp = 0
                    for c in range(k):
                        dcj = self.distance(X[j],P[c])
                        temp += pow(dij/dcj, 2 / (self.q - 1))
                    U[i][j] = 1/temp

            if iter%50 == 0:
                print('第%d次迭代' % iter)
                # print('P', P)
                # print('U',U)
                # print(np.sum(U[:, :], axis=0))

        # print(U.shape)
        return U,P

    # 分簇，把xj分到可能性最大的簇里
    def clustering(self,X,k,max_iter=200):
        U,P = self.train(X,k,max_iter=max_iter)
        clusters = self.init_clusters(k)
        m,n = X.shape  # m个样本，n维
        for j in range(m):
            probability = list(U[:,j])
            temp = probability
            max_probability = sorted(temp,key=lambda x:x)[-1] # 最大的可能性
            index = probability.index(max_probability)  # 是第几个簇
            clusters[index].append(X[j]) # 分簇

        # 显示
        self.show_vectors(P)
        self.show_result(clusters)

    # 显示原型向量，就是中心点
    def show_vectors(self, vectors):
        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange', 'purple']
        for i in range(vectors.shape[0]):
            plt.scatter(vectors[i, 0], vectors[i, 1], c=colors[i], marker='*')

    # 显示分簇的结果
    def show_result(self, clusters):
        k = len(clusters)
        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange', 'purple']
        for i in range(k):
            ci = np.array(clusters[i])
            print(ci.shape)
            if ci != []:
                plt.scatter(ci[:, 0], ci[:, 1], c=colors[i], label='cluster')

        plt.title('FCM clustering')
        plt.legend()


if __name__ == "__main__":
    X,labels = create_data()
    inputs_sepal = X[:, :2]
    inputs_petal = X[:, 2:4]
    model = FCM()
    model.clustering(inputs_sepal,3)
    plt.show()
    # model.train(inputs_sepal,3)
    # model.init_divide_matrix(3,4)




