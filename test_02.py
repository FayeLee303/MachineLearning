"""PLA算法"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model(object):
    def __init__(self):
        # 初始化权重列表，n*1的矩阵，n是data数据集里x的维度，就是len(data[0])-1
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0
        self.learning_rate = 0.1

    def sign(self,x,w,b):
        y = np.dot(x,w) + b
        return y

    # 使用随机梯度下降法
    def fit(self,x_train,y_train):
        is_wrong = False # 是不是错点，遇到错点才更新
        # 没有遇到错点时循环
        while not is_wrong:
            wrong_count = 0
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]
                if y*self.sign(x,self.w,self.b) <= 0:
                    # 遇到错点进行更新
                    # w=w+r*y*x y*x是w的梯度
                    # b=b+r*y y是b的梯度
                    self.w = self.w + self.learning_rate * np.dot(y,x)
                    self.b = self.b + self.learning_rate * y
                    # print('fixed w:%s, fixed b:%s'%(self.w,self.b))
                    wrong_count += 1 # 修正了一个错点
                    # print(wrong_count)
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'


if __name__ == "__main__":
    # load data
    iris = load_iris()
    # 使用pandas解析数据
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df .columns = ['sepal length','sepal width','petal length','petal width','label']
    df.label.value_counts()

    # 画图
    plt.subplot(1,3,1)
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'],color='blue',label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'],color='orange',label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()

    # 将pandas存的数据取出来转换成numpyarray
    data = np.array(df.iloc[:100,[0,1,-1]])
    # -1什么意思？
    x_train,y_train = data[:,:-1], data[:,-1]
    y_train = np.array([1 if i==1 else -1 for i in y_train]) # 转成array

    preceptron = Model()
    preceptron.fit(x_train,y_train)

    plt.subplot(1, 3, 2)
    x_points = np.linspace(4,7,10)
    yy = -(preceptron.w[0]*x_points + preceptron.b) /preceptron.w[1]
    plt.plot(x_points,yy)

    plt.plot(data[:50,0],data[:50,1],'bo',color='blue',label='0')
    plt.plot(data[50:100,0],data[50:100,1],'bo',color='orange',label='1')
    plt.xlabel('sepal length')
    plt.plot('sepal width')
    plt.legend()

    # plt.show()

    # 使用sklearn自带的Perceptron
    clf = Perceptron(fit_intercept=False,max_iter=1000,shuffle=False)
    clf.fit(x_train,y_train)
    print(clf.coef_)    # 特征权重w
    print(clf.intercept_)   # 截距b
    x_points = np.arange(4,8)
    yy = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]

    plt.subplot(1,3,3)
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.plot('sepal width')
    plt.legend()

    plt.show()



