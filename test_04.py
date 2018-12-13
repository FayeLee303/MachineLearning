"""KDTree"""

import numpy as np
from math import sqrt
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import namedtuple

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
        self.rchild = rchild    # 分割超平面右子空间


"""KDT"""
class KdTree(object):
    def __init__(self,dataSet,depth=0):
        self.root = None # 根结点
        # 数据维度len(dataSet[0])
        self.k = np.shape(dataSet)[1]

        # self.nearest = None # 最近的点

        # 定义一个namedtuple分别存放最近坐标点，最近距离，访问过的结点数
        # namedtuple创建类似于元祖的数据类型，除了可以索引访问，还可以用属性名访问
        # 类似于C的结构体struct
        # self.result = namedtuple('Result_tuple', 'nearest_point nearest_dist nodes_visited')

    """
    构造KDT：输入数据集T，输出kd树
    构造根结点，根结点对应包含数据集T的k维空间的超矩形区域
    选择某个axis轴，以T中所有x在这个轴上的坐标的中位数为切分点，做垂直坐标轴的超平面把根结点对应的超矩形区域划分为两块
    由根结点生成深度为1的左右自结点，左结点是坐标在axis轴的投影小于切分点的子区域，右结点同理
    落在切分超平面上的实例保存在根结点，就是在axis轴上投影刚好是中位数
    重复
        对深度为j的结点，选择第l维度的axis轴，l=j(mod k)+1
        所有的样本点在该轴上做投影，选择中位数为切分点
        生成深度为j+1的左右子结点，小左大右，落在轴上的保存在该结点
    直到两个子区域没有实例时停止
        
    """
    # 构造KDT
    def create(self,dataSet,depth=0):
        if len(dataSet) < 0:
            return None # 数据集为空
        else:
            # m, n = np.shape(dataSet) # m n是数据集的行列
            # self.n = n-1 # 去掉label一列
            # axis = depth % self.n   # l=j*mod(k)+1
            # mid = int(m/2)  # 样本点中间位置
            # # 根据关键字对dataSet里的数据进行排序
            # # 关键字是x[axis] 根据轴从X里取值？
            # dataSetCopy = sorted(dataSet, key = lambda x:x[axis])
            # # 落在轴上的归为根结点
            # node = Node(dataSetCopy[mid], depth)
            # if depth == 0:
            #     self.kdTree == node # 新结点为根结点
            # # 比中位数小的划分为左结点，比中位数大的划分为右结点
            # node.lchild = self.create(dataSetCopy[:mid],depth+1)
            # node.rchild = self.create(dataSetCopy[mid+1:],depth+1)
            # return node # 返回根结点

            # depth默认为0，每创建一次子树就+1
            # 进行分割的维度序号 l=j(mod k)+1,k是总维度，j是当前深度
            # 注意对一颗KD树，同一层的结点用于划分的维度轴是一样的！
            # 只是同一层的不同结点需要划分的中位数不一样
            # 作为划分依据的中位数就存在该结点中
            split = depth % self.k + 1  # 当前作为切分依据的轴是哪个维度的

            # 根据要切分的轴对数据进行排序
            dataSet.sort(key=lambda x:x[split])
            split_pos = len(dataSet) // 2 # 切分位置为中间位置
            median = dataSet[split_pos] # 拍完序就可以找到中位点

            node = Node(median,depth)
            if depth == 0:
                self.root = node # 新结点为根结点
            # 递归地创建kd树
            # 比中位数小的划分为左结点，比中位数大的划分为右结点
            node.lchild = self.create(dataSet[:median],depth+1)
            node.rchild = self.create(dataSet[median+1:],depth+1)
            return node

    # 先序遍历
    def preOrder(self,node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)


    """
    搜索最近邻：输入构造好的KDT和目标点x,输出x的最近邻点
    在kdt里找包含目标结点x的叶子结点：
        从根结点出发，递归地向下访问，如果目标x比切分点当前维度的坐标小，就往左子树，大就往右子树，直到子结点为叶子结点
    以该叶子结点当作当前最近点
    递归地向上回退，对每个结点
        如果该结点保存的样本点比当前最近点和目标点的距离更近，就更新当前最近点
        当前最近点一定存在与该结点的一个子结点对应区域，检查该子结点的父结点的另一个子结点（兄弟结点）对应的区域是否有更近的点
        即以目标点为圆心，目标点和当前最近点距离为半径作圆，是否和该结点的兄弟结点的区域相交
        如果相交，就在兄弟结点的区域里找更近的点，即当前结点=当前结点的兄弟结点
        如果不相交，进行回退
    直到回退到根结点搜索结束，最后的当前最近点就是目标点x的最近邻点        
    """

    def search(self,dataSet,x_point):
        node = self.root
        stack = [node] # 用来回退的栈

        # 直到找到node是叶子结点
        while node.lchild is not None or node.rchild is not None:
            split = node.depth % self.k + 1  # 当前作为切分依据的轴是哪个维度的
            # 根据要切分的轴对数据进行排序
            dataSet.sort(key=lambda x: x[split])
            split_pos = len(dataSet) // 2  # 切分位置为中间位置
            median = dataSet[split_pos]  # 拍完序就可以找到中位点
            if x_point[split] < median:
                node = node.lchild
            elif x_point[split] > median:
                node = node.rchild
            stack.append(node) # 入栈

        nearest_node = node # 先将叶子结点当作当前最近点

        # 回退，直到根结点时搜索结束
        while stack:
            node = stack.pop() # 取一个结点
            # 计算该结点保存的样本点和目标点的距离
            dist_now = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x_point, node.data)))
            # 计算当前最近点和目标点的距离
            dist_near = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x_point, nearest_node.data)))
            # 如果当前结点距离更近，就更新当前最近点
            if dist_now < dist_near:
                nearest_node = node
            else:
                # 取出父结点
                node_father = stack.pop()
                # 找到兄弟结点
                if node is node_father.lchild:
                    node_brother = node_father.rchild
                else:
                    node_brother = node_father.lchild
                # 检查兄弟结点对应的区域是否有更近的点
                # 即以目标点为圆心，目标点和当前最近点距离为半径作圆，是否和兄弟结点的区域相交
                # 计算当前最近点和目标点的距离
                dist_near = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x_point, nearest_node.data)))


        # # 遍历KDT
        # def travel(self,kd_node,target,max_dist):
        #     if kd_node is None:
        #         # python中用float("inf")和float("-inf")表示正负无穷
        #         return self.result([0]*self.k,float('inf'),0)
        #     nodes_visited = 1
        #     split = kd_node.depth % self.k+1 # 进行分割的维度
        #     pivot = kd_node.data[split] # 进行分割的轴
        #     if target[split] < pivot:   # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
        #         nearer_node = kd_node.lchild # 下一个访问节点为左子树根节点
        #         further_node = kd_node.rchild   # 同时记录下右子树
        #     else:
        #         nearer_node = kd_node.rchild  # 下一个访问节点为右子树根节点
        #         further_node = kd_node.lchild  # 同时记录下左子树
        #
        #     # 进行遍历找到包含目标点的区域
        #     temp1 = travel(nearer_node,target,max_dist)
        #     # 以此叶结点作为“当前最近点”
        #     nearest = temp1.nearest_point
        #     # 更新最近距离
        #     dist = temp1.nearest_dist
        #     # 更新已经访问过的点
        #     nodes_visited += temp1.nearest_point
        #
        #     if dist < max_dist:
        #         # 最近点将在以目标点为球心，max_dist为半径的超球体内
        #         max_dist = dist
        #
        #     # 第split维上目标点与分割超平面的距离
        #     temp_dist = abs(pivot - target[split])
        #     # 判断超球体是否与超平面相交
        #     if max_dist < temp_dist:
        #         # 不相交则可以直接返回，不用继续判断
        #         return self.result(nearest,dist,nodes_visited)
        #
        #     # 计算目标点和分割点的欧式距离
        #     temp_dist = sqrt(sum((p1 -p2)**2 for p1,p2 in zip(pivot,target)))
        #     if temp_dist < dist:
        #         # 如果更近，就更新最近点
        #         nearest = pivot
        #         # 更新最近距离
        #         dist = temp_dist
        #         # 更新超球体半径
        #         max_dist = dist
        #
        #     # 检查另一个子结点对应的区域是否有更近的点
        #     temp2 = travel(further_node, target, max_dist)
        #     nodes_visited += temp2.nodes_visited
        #     if temp2.nearest_dist < dist:
        #         nearest = temp2.nearest_point
        #         dist = temp2.nearest_point
        #
        #     return self.result(nearest,dist,nodes_visited)
        # # return travel(tree.root,x_point,float('inf'))