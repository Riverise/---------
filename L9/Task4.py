import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import Any
from L9.Task1 import PLA, Pocket_PLA

np.random.seed(45)

def generate_dataset(mean1, mean2, cov1, cov2, num):
    data_pos = np.random.multivariate_normal(mean1, cov1, num)
    data_neg = np.random.multivariate_normal(mean2, cov2, num)
    X = np.concatenate([data_pos, data_neg], axis=0)
    label_pos = np.array([1] * num)
    label_neg = np.array([-1] * num)
    y = np.concatenate([label_pos, label_neg], axis=0)
    return X, y


def split_dataset(X: np.ndarray[Any, Any], y: np.ndarray[Any], test_size: float):
    shuffled_idx = np.random.permutation(len(X))
    shuffled_X = X[shuffled_idx]
    shuffled_y = y[shuffled_idx]
    # split_dataset
    train_size = int(len(X) * (1 - test_size))
    train_X = shuffled_X[:train_size]
    train_y = shuffled_y[:train_size]
    test_X = shuffled_X[train_size:]
    test_y = shuffled_y[train_size:]
    return train_X, train_y, test_X, test_y


def get_acc(X: np.ndarray[Any, Any], y: np.ndarray[Any], w: np.ndarray[Any]):
    aug_X = np.insert(X, 0, 1, axis=1)
    temp = aug_X @ w
    temp2 = temp * y
    correct = sum(temp2 > 0)
    acc = correct / len(X)
    return acc


def visual(X, y, w1, w2):
    # 绘制数据点
    plt.figure(figsize=(8, 6))  # 设置图形的大小

    # 根据标签绘制不同的符号和颜色
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], s=4, c='red', marker='x', label='1' if i == 0 else "")  # 红色叉叉表示标签为1
        else:
            plt.scatter(X[i, 0], X[i, 1], s=4, c='blue', marker='o', label='-1' if i == 0 else "")  # 蓝色圈圈表示标签为-1
    # 绘制分界面
    plt.ylim(-3, 3)
    x_ = np.linspace(-5, 5, 400)
    y1 = (-w1[1] / w1[2]) * x_ + (w1[0] / w1[2])
    plt.plot(x_, y1, label='Decision Boundary of PLA', c='red', linestyle='--')

    y2 = (-w2[1] / w2[2]) * x_ + (w2[0] / w2[2])
    plt.plot(x_, y2, label='Decision Boundary of Pocket', c='green', linestyle=':')
    # 添加图例
    plt.legend(title="Label")
    # 添加标题和轴标签
    plt.title('Scatter Plot of Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')


m1 = np.array([1, 0])
m2 = np.array([0, 1])
cov1 = np.eye(2)
cov2 = np.eye(2)
one_sample_num = 200

X, y = generate_dataset(m1, m2, cov1, cov2, one_sample_num)
train_X, train_y, test_X, test_y = split_dataset(X, y, 0.2)

w0 = np.zeros(X.shape[1] + 1)  # X has not been augmented yet!
epoch_num = 3
lr = 0.25

start_time1 = time()
w1 = PLA(train_X, train_y, w0, epoch_num, lr=lr)
end_time1 = time()
duration1 = end_time1 - start_time1

start_time2 = time()
w2 = Pocket_PLA(train_X, train_y, w0, epoch_num, lr=lr)
end_time2 = time()
duration2 = end_time2 - start_time2

acc_train1 = get_acc(train_X, train_y, w1)
acc_test1 = get_acc(test_X, test_y, w1)
acc_train2 = get_acc(train_X, train_y, w2)
acc_test2 = get_acc(test_X, test_y, w2)

#print("Acc on train-set by PLA:", acc_train1)
print("Acc on test-set by PLA:", acc_test1)
#print("Acc on train-set by Pocket:", acc_train2)
print("Acc on test-set by Pocket:", acc_test2)

print("Run time of PLA:", duration1)
print("Run time of Pocket:", duration2)

visual(X, y, w1, w2)
plt.show()
