import numpy as np
from typing import Any


def PLA(X: np.ndarray[Any, Any], y: np.ndarray[int], w0: np.ndarray[Any], epoch_num: int, lr=0.1) -> np.ndarray[Any]:
    # augmentation
    X = np.insert(X, 0, 1, axis=1)
    assert len(X[0]) == len(w0), "Dimension Mismatch!"
    for epoch in range(epoch_num):
        if_wrong = False
        for i in range(len(X)):

            if (w0 @ X[i]) * y[i] <= 0:
                w0 += y[i] * X[i] * lr
                if_wrong = True
        if ~if_wrong:
            break
    return w0


# def Pocket_PLA(X: np.ndarray[Any, Any], y: np.ndarray[int], w0: np.ndarray[Any], epoch_num: int, lr=0.1) -> np.ndarray[
#     Any]:
#     def train_one_epoch(X, y, w):
#         assert len(X[0]) == len(w), "Dimension Mismatch!"
#         error_list = []
#         for i in range(len(X)):
#             if (X[i] @ w) * y[i] <= 0:
#                 error_list.append(i)
#                 w += y[i] * X[i] * lr
#         return error_list, w
#
#     # augmentation
#     X = np.insert(X, 0, 1, axis=1)
#     pocket = X[np.random.choice(len(X))]
#     for epoch in range(epoch_num):
#         _, w = train_one_epoch(X, y, w0)
#         w0 = w  # 权向量更新
#         ls1, _ = train_one_epoch(X, y, w0)  # 测试当前wt+1的错误数
#         ls2, _ = train_one_epoch(X, y, pocket)  # 测试w_hat即pocket的错误数
#         if len(ls1) < len(ls2):
#             pocket = w0
#             if len(ls1) > 0:
#                 w0 = X[np.random.choice(ls1)]
#         if len(ls2) == 0:
#             break
#     return pocket
def Pocket_PLA(X: np.ndarray, y: np.ndarray, w0: np.ndarray, epoch_num: int, lr=0.1) -> np.ndarray:
    """
    Pocket PLA algorithm.

    Parameters:
        X (np.ndarray): Input features. Shape (n_samples, n_features).
        y (np.ndarray): Labels. Shape (n_samples,).
        w0 (np.ndarray): Initial weight vector. Shape (n_features,).
        epoch_num (int): Number of epochs (iterations).
        lr (float): Learning rate. Default is 0.1.

    Returns:
        np.ndarray: Final weight vector.
    """
    w = np.copy(w0)  # Initialize weight vector
    w_pocket = np.copy(w)  # Initialize pocket weight vector
    X = np.insert(X, 0, 1, axis=1)
    min_error = np.inf  # Initialize minimum error

    for _ in range(epoch_num):
        error = 0
        for i in range(X.shape[0]):
            if y[i] * np.dot(X[i], w) <= 0:
                w += lr * y[i] * X[i]  # Update weight vector

                # Calculate error
                error_i = np.sum(np.sign(np.dot(X, w)) != y)
                if error_i < min_error:
                    min_error = error_i
                    w_pocket = np.copy(w)

                error += 1

        # If converged, exit
        if error == 0:
            break

    return w_pocket

