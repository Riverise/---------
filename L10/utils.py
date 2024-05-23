import numpy as np
from typing import Any


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
    y_pred = np.sign(temp)
    correct = y_pred == y
    return np.sum(correct) / len(X)