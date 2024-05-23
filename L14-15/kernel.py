import numpy as np

'''
核函数功能
根据输入x1, x2，计算核函数值

若x1, x2包含多个样本
x1: (n, d)
x2: (m, d)
计算结果为: (n, m)
结果的每个元素为: K(x1[n], x2[m])
'''

def gaussian_kernel(x, y, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def polynomial_kernel(x1, x2, zeta=1, gamma=0.5, order=4):
    return (zeta + gamma * np.dot(x1, x2)) ** order
