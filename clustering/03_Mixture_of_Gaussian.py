"""高斯混合聚类，使用概率模型"""
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
输入 样本集D，m个样本，每个样本n维，不需要标签
    高斯混合成分个数k
输出 簇划分C={C1,C2..Ck}
过程
    初始化高斯混合分布的模型参数{ai,ui,Ei}i属于[1,k]
    其中，ai是第i个混合成分的概率
        u是n维的均值向量，
        E是n*n的协方差矩阵
    重复
        for j in range(m):
            计算xj由各混合成分生成的后验概率
        for i in range(k):
            计算新的均值向量new_ui
            计算新的协方差矩阵new_Ei
            计算新的混合系数new_ai
        更新ai,ui,Ei
    直到满足停止条件
    初始化C为空
    for j in range(m):
        根据公式确定xj的簇标记tj
        将xj划分相应的簇里
"""
class MOG(object):
    def __init__(self):
        pass

    # 初始化簇集合
    def init_clusters(self,k):
        clusters = {}
        for i in range(k):
            clusters[i] = []
        return clusters

    # 初始化alpha矩阵，概率矩阵
    # ai是第i个混合成分的概率
    # 有m个样本
    def init_alpha(self,m):
        return np.zeros((m,1))

    # u是n维的均值向量
    def init_mean_vector(self,n):
        return np.zeros((1,n))

    # E是n*n的协方差矩阵
    def init_covariance_matrix(self,n):
        return np.zeros((n,n))

    def train(self,X,k,max_iter=200):
        m,n = X.shape   # m个样本,n维
        # 初始化模型参数
        alpha = self.init_alpha(m)
        mean_vector = self.init_mean_vector(n)
        cov_matrix = self.init_covariance_matrix(n)

        # 初始化簇
        cluster = self.init_clusters(k)

        # 迭代
        for temp in range(max_iter):
            # 后验概率矩阵，每个xj对应一个后验概率
            post_probability_matrix = np.zeros((m,1))
            for j in range(m):
                # post_probability_matrix[j] =
                pass

            # 计算新的参数
            for i in range(k):
                # new_ai =
                # new_Ei =
                # new_ui =
                pass
            # 更新
            # ai = new_ai
            # Ei = new_Ei
            # ui = new_ui


        # 分簇
        for j in range(m):
            # 把xj分簇
            pass

        # 显示
        self.show_result()

    def show_result(self):
        pass

if __name__ == '__main__':
    X, labels = create_data()