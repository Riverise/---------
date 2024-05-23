import numpy as np
from cvxopt import solvers,matrix
from typing import Callable

class Primal_SVM:
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray):
        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.X = self.augment(X)
        self.y = y

    def quadratic_programming(self):
        Q = np.eye(1 + self.dim)
        Q[0, 0] = 0

        p = np.zeros((1 + self.dim, 1))

        # 因为cvxopt的不等式约束是 a^TU <= c ，所以加负号把不等式反过来
        a = -self.y.reshape((320,1)) * self.X
        c = -np.ones((self.n, 1))

        Q = matrix(Q)
        p = matrix(p)
        a = matrix(a)
        c = matrix(c)

        self.w = solvers.qp(Q, p, a, c)['x']
        self.w = np.array(self.w).squeeze() # 将[[x,y]]压缩成[x,y]
        return self.w

    def gradient_descent(self, lr:float=0.3, epochs:int=3, batch_size:int=1):
        self.w = np.zeros(1 + self.dim)
        data = np.concatenate((self.X, self.y), axis=1)
        np.random.shuffle(data)
        for epoch in range(epochs):
            for i in range(0, self.n, batch_size):
                batch = data[i : min(i + batch_size, self.n), :]
                batch_X = batch[:, :-1]
                batch_y = batch[:, -1]
                grad = self.hinge_loss_grad(self.w, batch_X, batch_y)
                self.w = self.w - lr * grad
            if np.all(1 - np.multiply(self.y, (self.x @ self.w)) <= 0):
                print('break at epoch {}'.format(epoch))
                break
        return self.w

    def hinge_loss(self, w, x, y):
        return np.max(0, 1 - np.multiply(y, (x @ w)))
    
    def hinge_loss_grad(self, w, x, y):
        batch_size = x.shape[0]
        condition = np.zeros(y.shape)
        condition[1 - np.multiply(y, (x @ w)) > 0] = 1
        grads = np.multiply(condition, np.multiply(-y, x))
        return np.sum(grads.T, axis=1) / batch_size
    
    def predict(self, test_X: np.ndarray):
        test_X_augmented = self.augment(test_X)  # 增加一列1，以便于表示偏置项b
        y_pred = np.sign(test_X_augmented @ self.w)
        return y_pred

    
    @staticmethod
    # 标记该方法为静态方法。静态方法不需要类实例即可调用，也不会自动传递实例,主要用于实现与类功能相关但在逻辑上不依赖于类实例的功能。
    def augment(X: np.ndarray) -> np.ndarray:
        X = np.insert(X,0,1,axis=1)
        return X
    
class Dual_SVM:
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray):
        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y.reshape((-1,1))

    def quadratic_programming(self):
        # 矩阵必须是np.ndarray类，若为np.matrix类，*不是哈达马积而是矩阵乘法
        Q = (self.X @ self.X.T) * (self.y @ self.y.T)
        p = -np.ones((self.n, 1))
        A = -np.eye(self.n)
        c = np.zeros((self.n, 1))
        # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # cvxopt中，rx直接相乘，不转置。因此先把它转置了
        r = self.y.T
        v = np.zeros((1, 1))

        Q = matrix(Q)
        p = matrix(p)
        A = matrix(A)
        c = matrix(c)
        r = matrix(r)
        v = matrix(v)

        # 注意这里 r 和 v 的维度应当分别是 (1, n) 和 (1, 1)
        r = matrix(self.y.T.astype(np.double))
        v = matrix(np.zeros((1, 1)).astype(np.double))

        alpha = solvers.qp(Q, p, A, c, r, v)['x']
        alpha = np.array(alpha)

        mask = np.where(alpha>1e-6)[0]
        self.sv_alpha = alpha[mask]
        self.sv_X = self.X[mask]
        self.sv_y = self.y[mask]

        self.w = (alpha * self.y).T @ self.X
        self.w = self.w[0, :]

        # 找到第一个不为0的alpha对应的样本序号
        # idx = np.where(alpha>1e-9)[0][0]
        b = self.y[mask[0]] - self.w @ self.X[mask[0]].T

        self.w = np.concatenate((b, self.w))
        return alpha

    def predict(self, test_X: np.ndarray):
        test_X_augmented = self.augment(test_X)  # 增加一列1，以便于表示偏置项b
        y_pred = np.sign(test_X_augmented @ self.w)
        return y_pred

    @staticmethod
    # 标记该方法为静态方法。静态方法不需要类实例即可调用，也不会自动传递实例,主要用于实现与类功能相关但在逻辑上不依赖于类实例的功能。
    def augment(X: np.ndarray) -> np.ndarray:
        X = np.insert(X,0,1,axis=1)
        return X
    
class Dual_Kernel_SVM:
    def __init__(self, X: np.ndarray, y: np.ndarray, kernel_func: Callable):
        self.dim = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y.reshape((-1, 1))
        self.kernel_func = kernel_func

    def quadratic_programming(self):
        # 计算核矩阵
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i, j] = self.kernel_func(self.X[i], self.X[j])

        # 计算 Q 矩阵
        Q = np.outer(self.y, self.y) * K
        p = -np.ones((self.n, 1))
        A = -np.eye(self.n)
        c = np.zeros((self.n, 1))
        r = matrix(self.y.T.astype(np.double))
        v = matrix(np.zeros((1, 1)).astype(np.double))

        Q = matrix(Q)
        p = matrix(p)
        A = matrix(A)
        c = matrix(c)

        alpha = solvers.qp(Q, p, A, c, r, v)['x']
        alpha = np.array(alpha)

        # 找到所有支撑向量
        idx = np.where(alpha > 1e-6)[0]

        self.sv_alpha = alpha[idx]
        self.sv_x = self.X[idx]
        self.sv_y = self.y[idx]

        # 计算偏置 b
        self.b = self.sv_y[0] - np.sum(self.sv_alpha * self.sv_y * K[idx[0], idx])

    def predict(self, X):
        # 计算输入样本与支持向量之间的核矩阵
        K = np.zeros((X.shape[0], self.sv_x.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.sv_x.shape[0]):
                K[i, j] = self.kernel_func(X[i], self.sv_x[j])

        # 计算预测值
        wx = np.dot(K, self.sv_alpha * self.sv_y)
        return np.sign(wx + self.b)

class Soft_Margin_Dual_Kernel_SVM(Dual_Kernel_SVM):
    def __init__(self, X: np.ndarray, y: np.ndarray, kernel_func: Callable, C: float):
        super().__init__(X, y, kernel_func)
        self.C = C

    def quadratic_programming(self):
        # 计算核矩阵
        K = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i, j] = self.kernel_func(self.X[i], self.X[j])

        # 计算 Q 矩阵
        Q = np.outer(self.y, self.y) * K
        p = -np.ones((self.n, 1))
        G_std = np.vstack((-np.eye(self.n), np.eye(self.n)))
        h_std = np.hstack((np.zeros(self.n), np.ones(self.n) * self.C)).reshape(-1, 1)
        r = matrix(self.y.T.astype(np.double))
        v = matrix(np.zeros((1, 1)).astype(np.double))

        Q = matrix(Q)
        p = matrix(p)
        G_std = matrix(G_std)
        h_std = matrix(h_std)
        r = matrix(self.y.T.astype(np.double))
        v = matrix(np.zeros((1, 1)).astype(np.double))

        alpha = solvers.qp(Q, p, G_std, h_std, r, v)['x']
        alpha = np.array(alpha)

        # 找到所有支撑向量
        idx = np.where(alpha > 1e-6)[0]

        self.sv_alpha = alpha[idx]
        self.sv_x = self.X[idx]
        self.sv_y = self.y[idx]

        # 计算偏置 b
        self.b = self.sv_y[0] - np.sum(self.sv_alpha * self.sv_y * K[idx[0], idx])


