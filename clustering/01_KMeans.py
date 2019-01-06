"""k均值算法，没有使用类标签"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import random
import math

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 读取数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:,:])
    return data[:,:-1], data[:,-1] # X Y

# 绘制sepal 2D图
def show_iris_sepal(data):
    plt.scatter(data[:50,0], data[:50,1], c='red', label='0')
    plt.scatter(data[50:100, 0], data[50:100, 1], c = 'green', label = '1')
    plt.scatter(data[100:, 0], data[100:, 1], c='blue', label='2')

    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('original')
    plt.legend()
    # plt.show()

# 绘制petal 2D 图
def show_iris_petal(data):
    plt.scatter(data[:50, 2], data[:50, 3], c='red', label='0')
    plt.scatter(data[50:100, 2], data[50:100, 3], c='green', label='1')
    plt.scatter(data[100:, 2], data[100:, 3], c='blue', label='2')

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('original')
    plt.legend()
    # plt.show()

# 绘制sepal 边界图
def show_sepal_boundary(inputs,labels):
    # 设置使用的颜色colors， 这里假设最后的结果是三个类别
    cm_dark = ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
    cm_light = ListedColormap(['r', 'g', 'b'])


    x = inputs[:, :2]  # x-axis sepal length-width
    y = labels

    offset = 0.5  # 防止数据在图形的边上而加上的一个偏移量，设定一个较小的值即可
    x_min, x_max = x[:, 0].min() - offset, x[:, 0].max() + offset
    y_min, y_max = x[:, 1].min() - offset, x[:, 1].max() + offset

    h = 0.1   # 分类边缘精确度
    # xx yy是把图像分成了n个网格，每个格子里有一个点
    # 把这些点都送到分类器里，相同类别的点涂成相同颜色
    # 所以绘制大片区域颜色，其实是绘制了N多个点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    knn = KNeighborsClassifier()
    knn.fit(x, y)
    # np.c_[xx.ravel(), yy.ravel()] 是把网格状的N个点和其标签组合起来，成x,y形式送入分类器
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # 得到的结果又变成网格状，用于之后画图
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cm_dark)  # 绘制底色
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=cm_light)  # 绘制数据的颜色

    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# 绘制petal 边界图
def show_petal_boundary(inputs,labels):
    # 设置使用的颜色colors， 这里假设最后的结果是三个类别
    cm_dark = ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
    cm_light = ListedColormap(['r', 'g', 'b'])


    x = inputs[:, 2:4]  # x-axis petal length-width
    y = labels

    offset = 0.5  # 防止数据在图形的边上而加上的一个偏移量，设定一个较小的值即可
    x_min, x_max = x[:, 0].min() - offset, x[:, 0].max() + offset
    y_min, y_max = x[:, 1].min() - offset, x[:, 1].max() + offset

    h = 0.1   # 分类边缘精确度
    # xx yy是把图像分成了n个网格，每个格子里有一个点
    # 把这些点都送到分类器里，相同类别的点涂成相同颜色
    # 所以绘制大片区域颜色，其实是绘制了N多个点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    knn = KNeighborsClassifier()
    knn.fit(x, y)
    # np.c_[xx.ravel(), yy.ravel()] 是把网格状的N个点和其标签组合起来，成x,y形式送入分类器
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # 得到的结果又变成网格状，用于之后画图
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cm_dark)  # 绘制底色
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=cm_light)  # 绘制数据的颜色

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

"""
输入 样本集D有m个样本，聚类簇数k
输出 簇划分C={C1,C2...Ck}
过程
    从D中随机选择k个样本作为初始中心点
    重复
        令Ci = None
        for j in range(m):
            计算样本点xj和k个中心点的距离
            根据距离最近的确定xj的簇标记
            将xj分到相应的簇里
            注意要把Xj从原来的簇里删掉！！！！
        for i in range(k):
            计算新的中心点（均值向量）
            如果 发生变化就更新
    直到所有的均值向量都不更新        
"""
class KMeans(object):
    def __init__(self,k):
        self.k = k # 聚类簇数k
        self.clusters = None
        self.centers = None

    # 距离度量
    def distance(self, x1, x2):
        return math.sqrt(sum(pow(x1 - x2, 2)))

    # 初始化簇集合
    def init_clusters(self,X):
        # 初始化空的簇集合
        clusters = {}
        for i in range(self.k):
            clusters[i] = []

        m = X.shape[0]
        # 第一次分簇
        for j in range(m):
            distances = []
            # 计算样本点xj和所有中心点的距离
            for i in range(self.k):
                distances.append(self.distance(X[j], self.centers[i]))  # 计算距离
            nearest_distance = np.min(distances)  # 最近的距离
            nearest_index = distances.index(nearest_distance)  # 是第几个
            # 给xj标记属于哪个簇
            X[j] = list(X[j])
            # X[j].append(nearest_index)
            clusters[nearest_index].append(X[j])  # 分簇
        return clusters

    # 随机产生初始中心点
    def init_center(self, X):
        m,n = X.shape # m个样本，每个样本n维
        centers = np.zeros((self.k,n))

        for i in range(self.k):
            index = random.randint(0, m - 1)
            centers[i] = X[index]
        # temp = []
        # for i in range(self.k):
        #     index = random.randint(0,m-1)
        #     if len(temp) == 0:
        #         temp.append(index)
        #         centers[i] = X[index]
        #     else:
        #         # 如果有重复的点就重新抽
        #         while index in temp:
        #             index = random.randint(0, m - 1)
        #         temp.append(index)
        #         centers[i] = X[index]
        return centers

    # 求簇的中心点
    def new_center(self, cluster):
        cluster = np.array(cluster)
        return np.mean(cluster)

    # 分簇
    def clustering(self,centers,):
        # 用栈
        # for c in range(self.k):
        #     cluster = self.clusters[c]  # 当前簇
        #     # X = np.array(cluster)
        #     for j in range(len(cluster)-1):
        #         # 对每个簇的元素做遍历，如果有离更近的簇就分过去，否则就留下
        #         distances = []
        #         for i in range(self.k):
        #             distances.append(self.distance(cluster[j], centers[i]))  # 计算距离
        #         nearest_distance = np.min(distances)  # 最近的距离
        #         nearest_index = distances.index(nearest_distance)  # 是第几个簇
        #         if nearest_index == c:
        #             # 如果是当前簇就不做处理
        #             continue
        #         else:
        #             # 加到新簇里
        #             self.clusters[nearest_index].append(cluster[j])  # 分簇
        #             # 并把当前簇的这个样本点删掉
        #             # for x in self.clusters[c]:
        #             #     if (x==np.array(X[j])).all():
        #             #         self.clusters[c].remove(x)
        #             # print(self.clusters[c])
        #             # print(X[j])
        #             # print(X[j].all())
        #             # print(type(X[j]))
        #             # print(self.clusters[c].index(X[j].all()))
        #             # TODO
        #             # self.clusters[c].remove(X[j].all())
        #             # self.clusters[c].remove(X[j].all())
        #             # del (self.clusters[c])[j]
        #             print(type(cluster[j]))
        #             self.clusters[c].remove(cluster[j].all())
        return self.clusters

    # 更新中心点
    def update_centers(self):
        new_centers = np.zeros(self.centers.shape)
        for i in range(self.k):
            new_centers[i] = self.new_center(self.clusters[i])
        # 四舍五入
        new_centers = np.around(new_centers, 2)
        return new_centers

    # 训练
    def fit(self,X):
        self.centers = self.init_center(X)  # 初始中心点
        self.clusters = self.init_clusters(X) # 初始分簇
        flag = False
        while flag == False:
            # 更新
            new_centers = self.update_centers()
            # 分簇
            self.clustering(self.centers)

            # 中心点没变，表示已经分完簇了
            if (new_centers == self.centers).all():
                print('Done!')
                flag = True
            else:
                # 否则用新的中心点开始下一轮
                self.centers = new_centers

        # 显示
        self.show_result(self.clusters)
        self.show_vectors(self.centers)
        return self.clusters

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

if __name__ == "__main__":
    inputs, labels = create_data()
    inputs_sepal = inputs[:,:2]
    inputs_petal = inputs[:, 2:4]

    model = KMeans(3)

    clusters = model.fit(inputs_sepal)
    # clusters = model.fit(inputs_petal)
    plt.show()

    # a = [np.array([1,2]),[3,4]]
    # print(a)
    # print(type(a))
    # print(type(a[0]))
    #
    # # a.remove(a[0])
    # # a.remove(np.array([1,2]).all())
    # # print(a.index(np.array([1,2])))
    # for aa in a:
    #     if (aa == np.array([1,2])).all():
    #         a.remove(aa)
    # print(a)

    # a = np.array([1,2,3])
    # b = np.array([4,5,6])
    # # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    # # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    # print(np.c_[a,b])
    # print(np.r_[a,b])
    # print(np.hstack((a,b)))
    # print(np.vstack((a,b)))
    # print(a.ravel())

