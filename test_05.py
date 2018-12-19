"""朴素贝叶斯"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

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
    pass


if __name__ == '__main__':
    inputs, labels = create_data() # X Y
    x_train, x_test, y_train, y_test = train_test_split(inputs,labels,test_size=0.3)

