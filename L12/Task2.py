import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from Task1 import logistic_regression, sigmoid
from utils import generate_dataset, split_dataset, visual

np.random.seed(42)

m1 = np.array([-1, 0])
m2 = np.array([0, 2])
cov1 = np.eye(2)
cov2 = np.eye(2)
one_sample_num = 200

X, y = generate_dataset(m1, m2, cov1, cov2, one_sample_num)
train_X, train_y, test_X, test_y = split_dataset(X, y, 0.2)

w, losses = logistic_regression(train_X, train_y, lr=0.1, epoch_num=2, batch_size=25)

y_p = []
y_pred = []
test_X_aug = np.insert(test_X, 0, 1, axis=1)
for i in range(len(test_X)):
    s = w @ test_X_aug[i]
    p = sigmoid(s)
    y_p.append(p)
    predict = 1 if p > 0.5 else -1
    y_pred.append(predict)
y_pred = np.array(y_pred)
acc = sum(y_pred == test_y) / len(y_pred)
print(acc)
# decision boundary
visual(X, y, w)
# loss decay
plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title('Loss change along with epoch rise')
plt.xlabel('Epoch num')
plt.ylabel('Loss')
plt.show()