import numpy as np
from typing import Any, Tuple


def normal_equation(X: np.ndarray[Any, Any], y: np.ndarray[Any]) -> Tuple[np.ndarray[Any], float]:
    """X:[sample_num, feature_num],X @ w = y"""
    X = np.insert(X, 0, 1, axis=1)
    # w = inv(X.T @ X) @ X.T @ y
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    y_predict = X @ w
    loss = np.linalg.norm(y_predict - y)
    return w, loss


def gradient_descent(X: np.ndarray[Any, Any], y: np.ndarray[Any], lr=0.1, epoch_num=1000):
    X = np.insert(X, 0, 1, axis=1)
    w = np.random.randn(X.shape[1])
    losses = []
    for _ in range(epoch_num):
        loss = np.linalg.norm(X @ w - y)
        losses.append(loss)
        grad = X.T @ (X @ w - y) * 2 / len(X)
        w = w - lr * grad
    return w, losses
