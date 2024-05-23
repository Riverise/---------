import numpy as np
from typing import Any

def analyze(X: np.ndarray[Any, Any]):
    mean = np.mean(X, axis=0)
    d = X.shape[1]
    cov = np.zeros((d,d))
    for n in range(len(X)):
        temp = X[n] - mean
        cov = cov + np.outer(temp, temp)
    return mean, cov


def fisher(X: np.ndarray[Any, Any], y: np.ndarray[Any]):
    mask = y == 1
    pos_X = X[mask]
    neg_X = X[~mask]
    mean1, cov1 = analyze(pos_X)
    mean2, cov2 = analyze(neg_X)
    s = cov1 + cov2
    w = np.linalg.inv(s) @ (mean1 - mean2)
    threshold = w @ (mean1 + mean2) / 2
    return w, threshold
