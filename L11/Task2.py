import numpy as np
from L11.Task1 import fisher
from utils import generate_dataset, split_dataset, get_acc, visual

m1 = np.array([-5, 0])
m2 = np.array([0, 5])
cov1 = np.eye(2)
cov2 = np.eye(2)
one_sample_num = 200

X, y = generate_dataset(m1, m2, cov1, cov2, one_sample_num)
train_X, train_y, test_X, test_y = split_dataset(X, y, 0.2)

w, threshold = fisher(train_X, train_y)

acc_train = get_acc(train_X, train_y, w, threshold)  # build-in augmentation
acc_test = get_acc(test_X, test_y, w, threshold)

print("Acc on train-set:", acc_train)
print("Acc on test-set:", acc_test)

visual(X, y, w, threshold)
