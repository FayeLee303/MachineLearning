"""朴素贝叶斯"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

from sklearn.naive_bayes import GaussianNB  # 高斯分布模型
from sklearn.naive_bayes import BernoulliNB, MultinomialNB # 伯努利分布模型，多项式分布模型

# 读取数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100,:]) # 取前100个样本
    # print(data)
    return data[:,:-1],data[:,-1] # X Y


"""
朴素贝叶斯，假设样本的特征分布为高斯分布
概率密度函数p(xi|yk)=1/根号(2πσ^2)*exp(-(xi-μ)^2/2σ^2)

"""
class NaiveBayes(object):
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    # X是n维向量
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差是方差开根号
    def stdev(self,X):
        avg = self.mean(X) # 平均数
        # sqrt开平方根号 pow(x,y)计算x的y次幂
        return math.sqrt(sum([pow(xi-avg,2)for xi in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self,x,mean,stdev):
        exponent = math.exp(-(pow(x-mean,2)/(2*pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi)*stdev))*exponent

    # 处理X_train
    def summarize(self,train_data):
        summaries = [(self.mean(i),self.stdev(i))for i in zip(*train_data)]
        return summaries # 得到一个列表，存的是期望和标准差

    # 分类别求出数学期望和标准差
    def fit(self,X,y):
        labels = list(set(y)) # 存成列表形式
        data = {label:[] for label in labels}   # 存成字典形式
        for f,label in zip(X,y):
            data[label].append(f) # 把属于某类的数据放到一起
        # model是一个字典，存不同类别的数学期望和标准差
        self.model={label:self.summarize(value)for label,value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self,input_data):
        probabilities={} # 概率
        for label,value in self.model.items():
            probabilities[label] = 1 # 初始化
            for i in range(len(value)):
                mean,stdev = value[i]
                # 根据概率密度函数算出每个样本的概率
                # 每一类的概率 = 该类的所有样本的概率的累乘
                probabilities[label] *= self.gaussian_probability(input_data[i],mean,stdev)
        return probabilities

    # 预测
    def predict(self,X_test):
        # 计算样本的概率，从小到大排列，返回最大的概率对应的类
        label = sorted(self.calculate_probabilities(X_test).items(),key=lambda x:x[-1])[-1][0]
        return label

    # 准确率
    def score(self,X_test,y_test):
        right = 0
        for x,y in zip(X_test,y_test):
            label = self.predict(x)
            if label == y:
                right += 1
        print('accuracy rate:',right / float(len(x_test)))
        return right / float(len(x_test))



if __name__ == '__main__':
    inputs, labels = create_data() # X Y
    x_train, x_test, y_train, y_test = train_test_split(inputs,labels,test_size=0.3)
    model = NaiveBayes()
    model.fit(x_train,y_train) # 训练
    print(model.predict([4.4,3.2,1.3,0.2]))
    model.score(x_test,y_test) # 预测准确率

    # 使用sklearn自带的函数
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    clf.score(x_test,y_test)
    clf.predict([[4.4,3.2,1.3,0.2]])

