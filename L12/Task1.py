import numpy as np
from typing import Any

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(X: np.ndarray[Any, Any], y: np.ndarray[Any], lr=0.1, epoch_num=50, batch_size=25):
    X = np.insert(X, 0, 1, axis=1)
    w = np.random.randn(X.shape[1])
    losses = []
    batch_num = int(len(X) / batch_size)
    for _ in range(epoch_num):
        loss = 0
        for batch_id in range(batch_num):
            grad = np.zeros_like(w) 
            for i in range(batch_id * batch_size, (batch_id+1) * batch_size):
                loss = loss + np.log(1 + np.exp(-1 * y[i] * (w @ X[i])))
                grad = grad + sigmoid(-1 * y[i] * (w @ X[i])) * (-1 * y[i] * X[i])
            grad = grad / batch_size
            w = w - lr * grad
        loss = loss / batch_size
        losses.append(loss) 
    return w, losses
