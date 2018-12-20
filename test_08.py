"""支持向量机SMO算法"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.svm import SVC


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1 # 把原数据里的0改成-1
    # print(data)
    return data[:, :2], data[:, -1] # 返回x,label


"""使用非线性可分分类器"""
"""
SMO算法
输入训练数据集，二分类，精度e
输出近似解alpha'
    取初始值alpha=0,k=0
    选取最优变化量alpha1_k,alpha2_k
        alpha1:外层循环找违反KKT条件最严重的样本点
            首先遍历所有满足0<aplhai<C的样本点，如果都满足就遍历整个训练集                
        alpha2:内层循环，找使|E1=E2|最大的点
            启发式
    求解两个变量的最优化问题，求得最优解alpha1_k+1,alpha2_k+1
    更新alpha = alpha_k+1
    更新Ei保存在列表里
    if在精度e范围内满足停止条件
        取alpha' = alpha_k+1
    else
        k = k+1
        重复
"""

class SVM(object):
    def __init__(self,max_iter=100,kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel # 内部访问

    # 初始化变量
    def init_args(self,features,labels):
        self.m, self.n = features.shape # m*n的图？
        self.X = features # 输入特征
        self.Y = labels # 标签
        self.b = 0.0 # 截距

        self.alpha = np.ones(self.m) # 初始化拉格朗日乘子
        self.E = [self._E(i) for i in range(self.m)] # 把Ei保存在一个列表里
        self.C = 1.0 # 松弛变量

    # KKT条件
    """
    g(X[i]) = sum(alpha[i]*Y[i]*K(X,X[i])) + b
    alpha[i]=0 等价于 Y[i]*g(X[i]) >=1 分对的？
    0<alpha[i]<C 等价于 Y[i]*g(X[i]) =1 在间隔和超平面之间
    alpha[i]=C 等价于 Y[i]*g(X[i]) <=1 分错的？
    """
    def _KKT(self,i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # w是唯一的，b是不唯一的
    # g(x)预测值，输入xi，输出预测的y
    def _g(self,i):
        # r = self.b
        r = 0
        for j in range(self.m): # 对所有的样本Xi
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i],self.X[j])
        return r + self.b

    # 核函数
    def kernel(self,x1,x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)]) # 内积
        elif self._kernel == 'ploy':
            # 多项式核函数k(x,z) = (x*z+1)的p次幂，对应p次多项式分类器
            # 决策函数是f(x)=sign(sum(alpha[i]*Y[i]*K(x[i],x[j]))+b)
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        elif self._kernel == 'gaussian':
            # 高斯核函数K(x,z)=exp(-(x-z的范数)/2*方差)
            # 决策函数f(x)=sign(sum(alpha[i]*Y[i]*K(x[i],x[j]))+b)
            mean = sum(self.X) / float(len(self.X))
            variance = math.sqrt(sum([pow(xi-mean,2)for xi in self.X]) / float(len(self.X)))
            return math.exp(sum(-np.linalg.norm(x1[k],x2[k],ord=2)for k in range(self.n)) / 2*variance)


    # E(x)为g(x)对输入x的预测值和y的差，可以理解成损失
    def _E(self,i):
        return self._g(i) - self.Y[i]


    # SMO算法在每个子问题中选择两个变量进行优化
    # 固定其他变量不变
    def _init_alpha(self):
        # 外层循环，首先遍历所有满足0<alpha<C的样本点，检查是否满足KKT条件
        index_list = [i for i in range(self.m) if 0<self.alpha[i]<self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list) # 得到了所有的训练集样本点
        for i in index_list:
            if self._KKT(i):
                continue # 直到找到不满足KKT条件的点
            E1 = self.E[i]
            # 第一轮把alpha1确定，E1随之确定
            # alpha2使得|E1-E2|最大
            # 如果E2是正的，就选最小的，如果E2是负的，就选最大的
            # 体会利用lamba排序
            if E1 >=0:
                j = min(range(self.m),key=lambda i:self.E[i])
            else:
                j = max(range(self.m),key=lambda i:self.E[i])
            return i,j # 就是要优化的两个变量alpha1 alpha2


    """
    不等式约束0<=alpha[i]<=C使得alpha1和alpha2在[0,C]*[0,C]的正方形里
    条件约束alpha1*y1 + alpha2*y2 =-sum(y[i]*alpha[i]) = 某个常数
    也就是说alpha1和alpha2在一条直线上
    用这条直线和正方形相交，得到两个交点L H
    超过的部分就取端点值，中间就取alpha
    """
    def _compare(self,_alpha,L,H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练
    def fit(self,features,labels):
        self.init_args(features,labels) # 初始化参数
        for t in range(self.max_iter):
            i1,i2 = self._init_alpha() # 要优化的两个变量的index
            # 找到直线和正方形相交的两个端点的值
            # 这是alpha取值的边界
            # 就是把alpha的取值范围从[0,C]缩小到了[L,H]
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # SMO最优化问题的子问题沿着约束方向的未经剪辑的解是alpha2_new_unc= alpha2_old+y2(E1-E2)/h
            # h = K11 + K22 - 2K12 = (φ(x1)-φ(x2)) 的范数！
            eta = self.kernel(self.X[i1],self.X[i1]) + \
                  self.kernel(self.X[i2],self.X[i2]) - \
                  2*self.kernel(self.X[i1],self.X[i2])
            if eta <= 0:
                # 求的范数距离应该是大于0的，如果不是就说明错了
                # 重新找alpha1 alpha2
                continue

            # 沿着约束方向的未经剪辑的解
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2-E1) / eta
            # 进行剪辑，就是限制在[0,C]范围之内
            alpha2_new = self._compare(alpha2_new_unc,L,H)
            # 根据alpha2_new来求alpha1_new
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2]-alpha2_new)

            # 计算阈值b
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1],self.X[i1]) * (alpha1_new-self.alpha[i1]) - \
                     self.Y[i2] * self.kernel(self.X[i2],self.X[i1]) * (alpha2_new-self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - \
                     self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            # alpha1_new alpha2_new同时满足在区间[0,C],b1_new = b2_new = b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # alpha1_new alpha2_new是0或者C，就是都落在端点上了
                # 选择b1_new b2_new的中点作为新的b
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2) # 更新Ei保存在列表中

        print('train done!')


    # 预测
    def predict(self,data):
        r = self.b
        for i in range(self.m):
            # 决策函数f(x) = self.alpha[i] * self.Y[i] * self.kernel(data,self.X[i])
            r += self.alpha[i] * self.Y[i] * self.kernel(data,self.X[i])
        return 1 if r >0 else -1


    # 准确率
    def score(self,x_test,y_test):
        right = 0
        for i in range(len(x_test)):
            result = self.predict(x_test[i]) # 做预测
            if result == y_test[i]:
                right += 1
        print('accuracy rate:',right / len(x_test))
        return right / len(x_test)


    # 权重
    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1,1)*self.X
        self.w = np.dot(yx.T,self.alpha)
        return self.w


# class SVM(object):
#     def __init__(self, max_iter=100, kernel='linear'):
#         self.max_iter = max_iter
#         self._kernel = kernel
#
#     def init_args(self, features, labels):
#         self.m, self.n = features.shape
#         self.X = features
#         self.Y = labels
#         self.b = 0.0
#
#         # 将Ei保存在一个列表里
#         self.alpha = np.ones(self.m)
#         self.E = [self._E(i) for i in range(self.m)]
#         # 松弛变量
#         self.C = 1.0
#
#     def _KKT(self, i):
#         y_g = self._g(i) * self.Y[i]
#         if self.alpha[i] == 0:
#             return y_g >= 1
#         elif 0 < self.alpha[i] < self.C:
#             return y_g == 1
#         else:
#             return y_g <= 1
#
#     # g(x)预测值，输入xi（X[i]）
#     def _g(self, i):
#         r = self.b
#         for j in range(self.m):
#             r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
#         return r
#
#     # 核函数
#     def kernel(self, x1, x2):
#         if self._kernel == 'linear':
#             return sum([x1[k] * x2[k] for k in range(self.n)])
#         elif self._kernel == 'poly':
#             return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
#
#         return 0
#
#     # E（x）为g(x)对输入x的预测值和y的差
#     def _E(self, i):
#         return self._g(i) - self.Y[i]
#
#     def _init_alpha(self):
#         # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
#         index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
#         # 否则遍历整个训练集
#         non_satisfy_list = [i for i in range(self.m) if i not in index_list]
#         index_list.extend(non_satisfy_list)
#
#         for i in index_list:
#             if self._KKT(i):
#                 continue
#
#             E1 = self.E[i]
#             # 如果E2是+，选择最小的；如果E2是负的，选择最大的
#             if E1 >= 0:
#                 j = min(range(self.m), key=lambda x: self.E[x])
#             else:
#                 j = max(range(self.m), key=lambda x: self.E[x])
#             return i, j
#
#     def _compare(self, _alpha, L, H):
#         if _alpha > H:
#             return H
#         elif _alpha < L:
#             return L
#         else:
#             return _alpha
#
#     def fit(self, features, labels):
#         self.init_args(features, labels)
#
#         for t in range(self.max_iter):
#             # train
#             i1, i2 = self._init_alpha()
#
#             # 边界
#             if self.Y[i1] == self.Y[i2]:
#                 L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
#                 H = min(self.C, self.alpha[i1] + self.alpha[i2])
#             else:
#                 L = max(0, self.alpha[i2] - self.alpha[i1])
#                 H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
#
#             E1 = self.E[i1]
#             E2 = self.E[i2]
#             # eta=K11+K22-2K12
#             eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
#                 self.X[i1], self.X[i2])
#             if eta <= 0:
#                 # print('eta <= 0')
#                 continue
#
#             alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
#             alpha2_new = self._compare(alpha2_new_unc, L, H)
#
#             alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)
#
#             b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
#                 i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
#             b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
#                 i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b
#
#             if 0 < alpha1_new < self.C:
#                 b_new = b1_new
#             elif 0 < alpha2_new < self.C:
#                 b_new = b2_new
#             else:
#                 # 选择中点
#                 b_new = (b1_new + b2_new) / 2
#
#             # 更新参数
#             self.alpha[i1] = alpha1_new
#             self.alpha[i2] = alpha2_new
#             self.b = b_new
#
#             self.E[i1] = self._E(i1)
#             self.E[i2] = self._E(i2)
#         print('train done!')
#
#     def predict(self, data):
#         r = self.b
#         for i in range(self.m):
#             r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
#
#         return 1 if r > 0 else -1
#
#     def score(self, X_test, y_test):
#         right_count = 0
#         for i in range(len(X_test)):
#             result = self.predict(X_test[i])
#             if result == y_test[i]:
#                 right_count += 1
#         print('accuracy rate:', right_count / len(X_test))
#         return right_count / len(X_test)
#
#     def _weight(self):
#         # linear model
#         yx = self.Y.reshape(-1, 1) * self.X
#         self.w = np.dot(yx.T, self.alpha)
#         return self.w


if __name__ == "__main__":
    inputs,labels = create_data()
    x_train,x_test,y_train,y_test = train_test_split(inputs,labels,test_size=0.25)

    plt.scatter(inputs[:50, 0], inputs[:50, 1], label='0')
    plt.scatter(inputs[50:100, 0], inputs[50:100, 1], label='1')
    plt.legend()
    # plt.show()

    svm = SVM(max_iter=200)
    svm.fit(x_train,y_train)
    # svm.predict([4.4,3.2,1.3,0.2])
    svm.score(x_test,y_test)

    # # 使用sklearn自带的函数
    # clf = SVC()
    # clf.fit(x_train,y_train)
    # print(clf.score(x_test,y_test))