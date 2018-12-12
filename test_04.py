"""KDTree"""

import numpy as np
from math import sqrt
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['lable'] = iris.target # 标签列=iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100,[0,1,-1]])
train, test = train_test_split(data, test_size=0.1) # 选训练集和测试集
# 把样本按标签正负分成两堆
x0 = np.array([x0 for i,x0 in enumerate(train) if train[i][-1] == 0])
x1 = np.array([x1 for i,x1 in enumerate(train) if train[i][-1] == 1])

def show_train():
    plt.scatter(x0[:,0], x0[:,1], c='blue',label='0')
    plt.scatter(x1[:,0], x1[:,1], c='orange',label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')


"""KDT结点"""
class Node(object):
    def __init__(self,data,depth=0,lchild=None,rchild=None):
        self.data = data    # k维向量节点，k维空间中的一个样本点
        self.depth = depth  # 深度
        self.lchild = lchild    # 分割超平面左子空间
        self.rchild = rchild


"""KDT"""
class KdTree(object):
    def __init__(self):
        self.kdTree = None
        self.n = 0
        self.nearest = None # 最近的点

    # 构造KDT
    def create(self,dataSet,depth=0):
        if len(dataSet) >0:
            m, n = np.shape(dataSet) # m n是数据集的行列
            self.n = n-1 # 去掉label一列
            axis = depth % self.n   # l=j*mod(k)+1
            mid = int(m/2)  # 样本点中间位置
            # 根据关键字对dataSet里的数据进行排序
            # 关键字是x[axis] 根据轴从X里取值？
            dataSetCopy = sorted(dataSet, key = lambda x:x[axis])
            # 落在轴上的归为根结点
            node = Node(dataSetCopy[mid], depth)
            if depth == 0:
                self.kdTree == node # 新结点为根结点
            # 比中位数小的划分为左结点，比中位数大的划分为右结点
            node.lchild = self.create(dataSetCopy[:mid],depth+1)
            node.rchild = self.create(dataSetCopy[mid+1:],depth+1)
            return node # 返回根结点
        return None # 数据集为空

    # 先序遍历
    def preOrder(self,node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

