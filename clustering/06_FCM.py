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
    聚类簇数k, 迭代次数
输出 软划分矩阵U
    一组原型向量{p1...pk}，pi是个n维向量，其实就是每个簇的中心点
过程
    随机初始化软划分矩阵U，每个元素在0到1之间，每一列之和等于1
    初始化一组聚类原型向量
    重复直到达到停止停止条件
        计算原型向量P
        计算划分矩阵U       
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

        # 指标
        print('DBI',self.calculate_DBI(k,clusters,P))
        print('DI',self.calculate_DI(k,clusters))

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
                plt.scatter(ci[:, 0], ci[:, 1], c=colors[i], label='cluster%d'%i)

        plt.title('FCM clustering')
        plt.legend()

    # 计算簇间样本的平均距离
    def calculate_avg(self,cluster):
        X = np.array(cluster)
        m,n = X.shape   # m是该簇里的样本个数
        dist = np.zeros((m,m))  # 距离矩阵
        for i in range(m):
            for j in range(m):
                dist[i,j] = self.distance(X[i],X[j])
        sum = np.sum(dist)
        return 2/(m*(m-1)) * sum

    # 计算簇内样本的最远距离
    def calculate_diam(self,cluster):
        X = np.array(cluster)
        m, n = X.shape
        dist = np.zeros((m, m))  # 距离矩阵
        for i in range(m):
            for j in range(m):
                dist[i, j] = self.distance(X[i], X[j])
        return np.max(dist)

    # 计算簇Ci和簇Cj的最近样本（边缘）的距离
    def calculate_dmin(self,cluster_1,cluster_2):
        X1 = np.array(cluster_1)
        X2 = np.array(cluster_2)
        # m1是簇1的样本个数，m2是簇2的样本个数
        m1,m2 = X1.shape[0],X2.shape[0]
        dist = np.zeros((m1,m2))  # 距离矩阵
        for i in range(m1):
            for j in range(m2):
                dist[i, j] = self.distance(X1[i], X2[j])
        return np.min(dist)

    # 计算簇Ci和簇Cj的中心点p1p2的距离
    def calculate_dcen(self,i,j,P):
        return self.distance(P[i],P[j])

    # 计算DBI
    def calculate_DBI(self,k,clusters,P):
        temp1 = np.zeros((1, k))
        for i in range(k):
            temp2 = np.zeros((1, k))
            for j in range(k):
                if j==i:    # 算簇间中心点的距离dcen所以i=j时跳过
                    continue
                else:
                    temp2[:,j]  = (self.calculate_avg(clusters[i])+
                                   self.calculate_avg(clusters[j]))/\
                                  self.calculate_dcen(i,j,P)
            temp1[:,i] = np.max(temp2)
        sum = np.sum(temp1)
        return 1/k * sum

    # 计算DI
    def calculate_DI(self,k,clusters):
        temp1 = np.zeros((1, k))
        for i in range(k):
            # temp2 = np.zeros((1, k))
            temp2 = np.ones((1,k))  # 因为下面要求最小，如果初始化是0，就会都变成0
            for j in range(k):
                temp3 = np.zeros((1, k))
                for l in range(k):
                    temp3[:,l] = self.calculate_diam(clusters[l])
                if j == i:  # 算簇间dmin所以i=j时跳过
                    continue
                else:
                    temp2[:,j] = self.calculate_dmin(clusters[i],clusters[j]) / np.max(temp3)
            temp1[:,i] = np.min(temp2)
        return np.min(temp1)

if __name__ == "__main__":
    X,labels = create_data()
    inputs_sepal = X[:, :2]
    inputs_petal = X[:, 2:4]
    model = FCM(q=2)
    model.clustering(X,3,max_iter=200)
    # plt.show()




