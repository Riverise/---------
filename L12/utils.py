import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(mean1, mean2, cov1, cov2, num):
    data_pos = np.random.multivariate_normal(mean1, cov1, num)
    data_neg = np.random.multivariate_normal(mean2, cov2, num)
    X = np.concatenate([data_pos, data_neg], axis=0)
    label_pos = np.array([1] * num)
    label_neg = np.array([-1] * num)
    y = np.concatenate([label_pos, label_neg], axis=0)
    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float):
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

def visual(X, y, w):
    # 绘制数据点
    plt.figure(figsize=(8, 6))  # 设置图形的大小

    # 根据标签绘制不同的符号和颜色
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], s=4, c='red', marker='x', label='1' if i == 0 else "")  # 红色叉叉表示标签为1
        else:
            plt.scatter(X[i, 0], X[i, 1], s=4, c='blue', marker='o', label='-1' if i == 0 else "")  # 蓝色圈圈表示标签为-1
    # 绘制分界面

    x_ = np.linspace(-10, 10, 400)
    y1 = (-w[1] / w[2]) * x_ + (w[0] / w[2])
    plt.plot(x_, y1, label='Decision Boundary of LogitsticRegression', c='red', linestyle='--')

    # 添加图例
    plt.legend(title="Label")
    # 添加标题和轴标签
    plt.title('Scatter Plot of Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()