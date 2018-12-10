""""最小二乘法得到多项式函数线性回归"""
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数sin(2派x)
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
# numpy.poly1d([1,2,3])  生成  1x^2+2x^1+3x^0
# p是多项式的参数向量
def fit_func(p,x):
    f = np.poly1d(p)    # 根据参数向量生成多项式函数
    return f(x)

# 残差 多项式的输出和真实y的差距
def residuals_func(p,x,y):
    ret = fit_func(p,x) - y
    return ret

# 拟合
def fitting(m=0):
    # n是多项式的次数
    p_init = np.random.rand(m+1) # 随机初始化多项式参数向量
    # 最小二乘法
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    print('Fitting Parameters:',p_lsq[0])
    return p_lsq

# 正则项，惩罚项对抗过拟合
def residuals_fuc_regularization(p,x,y):
    regularization = 0.001
    ret = fit_func(p,x) - y
    # 使用L2范数作为正则化项
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
    return ret

# 拟合时加上正则项
def fitting_regularization(m=0):
    # n是多项式的次数
    p_init = np.random.rand(m + 1)  # 随机初始化多项式参数向量
    # 最小二乘法
    p_lsq_regularization = leastsq(residuals_fuc_regularization, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq_regularization[0])
    return p_lsq_regularization


if __name__ == "__main__":
    # 初始化十个点
    x = np.linspace(0,1,10)
    x_points = np.linspace(0,1,1000)
    # y是真实的目标函数的值
    y = real_func(x)
    # yy是加上正态分布的噪音的目标函数的值
    yy = [np.random.normal(0,0.1)+yyi for yyi in y]
    # 最小二乘法得到的系数
    p_lsq = fitting()
    # M=0
    p_lsq_0 = fitting(m=0)
    # M=1
    p_lsq_1 = fitting(m=1)
    # M=3
    p_lsq_3 = fitting(m=3)
    # M=9
    p_lsq_9 = fitting(m=9) # 当M=9时，多项式曲线通过了每个数据点，造成了过拟合

    # 可视化
    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq_9[0], x_points),label='fitted curve')
    plt.plot(x,yy,'bo',label='noise')
    # plt.legend()
    # plt.show()

    # 使用正则项
    p_lsq_regularization = fitting_regularization(m=9)
    plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='fitted with regularization curve')
    plt.legend()
    plt.show()


