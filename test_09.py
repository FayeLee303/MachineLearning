"""adaboost提升算法"""
"""
组合模型是由多个基本模型构成的模型
组合模型的预测效果往往比任意一个基本模型要好
装袋bagging：每个基本模型是由从总体样本中随机抽样得到的不同数据集进行
    训练得到，通过重抽样得到不同训练数据集的过程叫做装袋
提升boost：每个基本模型训练时数据集采用不同权值，针对上一个基本模型分
    类错误的样本增加权重，使新的模型重点关注误分类样本
算法步骤：
    给每个训练样本分配权重，初始权重都是1/N
    针对带有权重的样本进行训练，得到模型Gm
    计算模型Gm的误分类率em
    计算模型Gm的系数alpha_m = 1/2*log(1-em/em)
    根据误分类率em和当前权重向量Wm更新权重向量Wm+1
    计算组合模型f(x)的误分类率f(x) = sum(alpha_m*Gm(xi))
    当组合模型的误分类率或迭代次数达到阈值停止迭代，否则回到步骤二
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length','sepal width','petal length','petal width','label']
    data = np.array(df.iloc[:100,[0,1,-1]])
    for i in range(len(data)):
        if data[i,-1]==0:
            data[i,-1] = 1
    return data[:,:2], data[:,-1]


class AdaBoost(object):
    # n_estimators弱学习器的最大迭代次数，过小容易欠拟合，过大容易过拟合
    # learning_rate是每个弱学习器的权重缩减系数
    def __init__(self,n_estimators=50,learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self,datasets,labels):
        self.X = datasets
        self.Y = labels
        self.M,self.N = datasets.shape

        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化权重
        self.weights = [1.0/self.M]*self.M
        # G(x)系数alpha
        self.alpha = []

    def _G(self,features,labels,weights):
        m = len(features)
        error = 100000.0 # 无穷大
        best_v = 0.0    # 阈值
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate

        direct, compare_array = None,None
        for i in range(1,int(n_step)):
            v = features_min + self.learning_rate * i
            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k]>v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])

                compare_array_nagetive = np.array([-1 if features[k]>v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                if weight_error < error:
                    # 更新
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    # 计算alpha
    def _alpha(self,error):
        return 0.5 * np.log((1-error)/error)

    # 规范化因子
    def _Z(self,weighs,a,clf):
        return sum([weighs[i]*np.exp(-1*a*self.Y[i]*clf[i]) for i in range(self.M)])

    # 权值更新
    def _w(self,a,clf,Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1*a*self.Y[i]*clf[i])/Z

    # G(x)的线性组合
    def _f(self,alpha,clf_sets):
        pass

    # 分类器，v是阈值
    def G(self,x,v,direct):
        if direct == 'positive':
            return 1 if x>v else -1
        else:
            return -1 if x>v else 1

    def fit(self,X,y):
        # 初始化参数
        self.init_args(X,y)
        # 迭代
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度，选择误差最小的
            for j in range(self.N):
                features = self.X[:,j]
                # 分类阈值，分类误差，分类结果
                v,direct,error,compare_array = self._G(features,self.Y,self.weights)

                if error < best_clf_error:
                    # 更新
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                if best_clf_error == 0:
                    break

            # 计算G(x)的系数alpha
            alpha = self._alpha(best_clf_error)
            self.alpha.append(alpha)
            # 记录分类器
            self.clf_sets.append((axis,best_v,final_direct))
            # 规范化因子
            Z = self._Z(self.weights,alpha,clf_result)
            # 权值更新
            self._w(alpha,clf_result,Z)


    # 预测
    def predict(self,features):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = features[axis]
            result += self.alpha[i] * self.G(f_input,clf_v,direct)
        # sign
        return 1 if result > 0 else -1

    # 准确率
    def score(self,X_test,y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        return right_count / len(X_test)




if __name__ == '__main__':
    X,y = create_data()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    plt.scatter(X[:50,0],X[:50,1],label='0')
    plt.scatter(X[50:,0],X[50:,1],label='1')

    clf = AdaBoost(n_estimators=10,learning_rate=0.2)
    clf.fit(X_train,y_train)
    print('准确率',clf.score(X_test,y_test))

    # 100次结果
    # result = []
    # for i in range(1,101):
    #     X,y = create_data()
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #     clf = AdaBoost(n_estimators=100, learning_rate=0.2)
    #     clf.fit(X_train, y_train)
    #     r = clf.score(X_test, y_test)
    #     result.append(r)

    # 使用sklearn自带
    # clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.5)
    # clf.fit(X_train,y_train)
    # print('')
    # print('准确率',clf.score(X_test,y_test))






