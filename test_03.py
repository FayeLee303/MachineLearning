"""K近邻算法KNN"""
import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier # KNN分类器

"""
p=1 曼哈顿距离，绝对值的最大值
p=2 欧氏距离，平方和的开方
p=inf 各个坐标距离的最大值
"""

# 距离度量，默认是欧氏距离
def distance(x,y,p=2):
    # x,y维度相同并且都大于1维
    if len(x) == len(y) and len(x)>1:
        sum = 0
        for i in range(len(x)):
            # (累加(|xi-yi|p次方))开p次方根号
            sum += math.pow(abs(x[i] - y[i]),p)
        return math.pow(sum, 1/p)
    else:
        return 0

"""
KNN：输入训练集，输入新的样本点xx，输出xx所属的类y
根据给定的距离度量，在训练T中找到与xx最近的k个点
k是设定好的值，是临近点的个数
涵盖这k个临近点的xx的邻域叫做Nk(xx)邻域，可以找到k个点加到列表里
对这个Nk邻域里的点，根据多数表决原则，哪一类的点数目最多，就把xx分到这个类
"""


class KNN(object):
    def __init__(self,x_train,y_train,n_neighbors=3,p=2):
        self.k = n_neighbors    # 临近点个数，就是k近邻的k值
        self.p = p  # 控制距离度量
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, xx):
        knn_list = [] # Nk邻域
        # 对周围临近的点计算距离
        # 怎么找周围最近的点？
        # 初始先找了数据集里最前面的k个点self.x_train[0:k,:]
        # 相当于对Nk邻域进行初始化
        for i in range(self.k):
            # 计算距离
            # dist = np.linalg.norm(x - self.x_train[i],ord=self.p)
            dist = distance(xx, self.x_train[i], self.p)
            # 将距离和该点的类别加到knn_list里
            knn_list.append((dist,self.y_train[i]))

        # print(knn_list)

        # 对数据集里所有的点进行遍历，来更新Nk邻域
        # 前k个点已经在knn_list里了，所以self.x_train[k+1:,:]里找
        # 找到某个点比前k个点与xx样本点距离更近就进行更新替换
        for i in range(self.k, len(self.x_train)):
            # 从knn_list里取出和xx距离最大的点
            max_index = knn_list.index(max(knn_list,key=lambda x :x[0]))
            # 计算self.x_train[i]和xx的距离
            dist = distance(xx, self.x_train[i], self.p)
            # 如果self.x_train[i]和xx的距离比knn_list里和xx距离最大的点的距离小
            # 就进行更新
            if dist < (knn_list[max_index])[0]:
                knn_list[max_index] = (dist,self.y_train[i])

        # 统计
        # 多数表决法，xi所在邻域Nk里的点哪个类别多，就将xi划分为这个类别
        # knn_list里存了dist和self.y_train[i]
        # 是把类别取出来存到了knn里
        knn = [k[-1] for k in knn_list]
        # print('knn',knn)
        # Counter是计数器，用于追踪值的出现次数
        # count_pairs是个字典，存了值和值出现的次数
        count_pairs = Counter(knn)
        print(count_pairs)
        # 把count_pairs排序，找到出现次数最多的，就把xx分到这个类
        # TODO
        # max_count = sorted(count_pairs,key=lambda x:x)[-1]
        r = sorted(count_pairs.items())
        print(r)
        max_count = r[-1]
        return max_count

    # 计算正确率
    def score(self,x_test,y_test):
        right_count = 0
        n = 10
        for x,y in zip(x_test,y_test):
            label = self.predict(x)
            # 预测正确
            if label == y:
                right_count += 1
        print("预测准确率：%.2f"%(right_count / len(x_test)))
        return right_count / len(x_test)


if __name__ == "__main__":
    # # 课本例子3.1
    # # 三个点
    # x1 = [1,1]
    # x2 = [5,1]
    # x3 = [4,4]
    # # 求p取不同值时，Lp距离下x1的最近邻点
    # for i in range(1,5):
    #     r = {'{}'.format(j):distance(x1,j,p=i)for j in [x2,x3]}
    #     print(min(zip(r.values(),r.keys())))

    # 加载数据
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    # # 画图
    # plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'],c='blue',label='0')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='red', label='1')
    # plt.xlabel('sepal length')
    # plt.ylabel('sepal width')
    # plt.legend()

    data = np.array(df.iloc[:100,[0,1,-1]])
    # -1理解为最后一列？
    # x取所有行和除了最后1列的所有列，y取所有行和最后1列
    x,y = data[:,:-1], data[:,-1]
    # train_test_split可以实现自动划分测试集训练集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    clf = KNN(x_train,y_train)
    # clf.score(x_test,y_test)

    # 对新来的样本点进行预测
    test_point = [5.3,3.2]
    print('test point:{}'.format(clf.predict(test_point)))

    # 画图
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='blue', label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='red', label='1')
    plt.plot(test_point[0],test_point[1],'bo',c='yellow',label='test point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    # # 使用sklearn自带的KNN分类器
    # clf_sk = KNeighborsClassifier()
    # clf_sk.fit(x_train,y_train)
    # clf.score(x_test,y_test)

