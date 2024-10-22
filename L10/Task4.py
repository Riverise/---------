import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from L10.Task1 import normal_equation, gradient_descent
from utils import generate_dataset, split_dataset, get_acc

np.random.seed(42)
m1 = np.array([4, 2])
m2 = np.array([0, 4])
cov1 = np.array([[2, 1],
                 [1, 2]])
cov2 = np.eye(2)
one_class_num = 75

X, y = generate_dataset(m1, m2, cov1, cov2, one_class_num)
train_X, train_y, test_X, test_y = split_dataset(X, y, 0.2)

w1, loss = normal_equation(train_X, train_y)  # build-in augmentation
w2, losses = gradient_descent(train_X, train_y, lr=0.005, epoch_num=1000)

acc_train1 = get_acc(train_X, train_y, w1)  # build-in augmentation
acc_test1 = get_acc(test_X, test_y, w1)
acc_train2 = get_acc(train_X, train_y, w2)
acc_test2 = get_acc(test_X, test_y, w2)

# print("Acc on train-set by NE:", acc_train1)
print("Acc on test-set by NE:", acc_test1)
# print("Acc on train-set by GD:", acc_train2)
print("Acc on test-set by GD:", acc_test2)

print(losses)

# visualization
plt.figure(figsize=(12, 6))  # 设置图形的大小

plt.subplot(121)
plt.xlim(-3, 8)
plt.ylim(-3, 8)
# 根据标签绘制不同的符号和颜色
for i in range(len(X)):
    if y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], s=4, c='red', marker='x', label='1' if i == 0 else "")  # 红色叉叉表示标签为1
    else:
        plt.scatter(X[i, 0], X[i, 1], s=4, c='blue', marker='o', label='-1' if i == 0 else "")  # 蓝色圈圈表示标签为-1
# 绘制分界面

x_ = np.linspace(-10, 10, 400)
y1 = (-w1[1] / w1[2]) * x_ + (w1[0] / w1[2])
plt.plot(x_, y1, label='Decision Boundary of Normal-Equation', c='red', linestyle='--')

y2 = (-w2[1] / w2[2]) * x_ + (w2[0] / w2[2])
plt.plot(x_, y2, label='Decision Boundary of Gradient-Descent', c='green', linestyle=':')
# 添加图例
plt.legend(title="Label")
# 添加标题和轴标签
plt.title('Scatter Plot of Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(122)
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel("loss")

plt.show()
