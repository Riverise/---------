import numpy as np
from kernel import gaussian_kernel, polynomial_kernel
from utils import generate_dataset, split_dataset, visual, acc_estimator, acc_estimator2
from Task1 import Primal_SVM, Dual_SVM, Dual_Kernel_SVM, Soft_Margin_Dual_Kernel_SVM
from cvxopt import solvers

np.random.seed(45)

solvers.options['show_progress'] = False

m1 = np.array([-5, 0])
m2 = np.array([0, 5])
cov1 = np.eye(2)
cov2 = np.eye(2)
one_sample_num = 200

X, y = generate_dataset(m1, m2, cov1, cov2, one_sample_num)
train_X, train_y, test_X, test_y = split_dataset(X, y, 0.2)

# Primal SVM
svm = Primal_SVM(train_X, train_y)
svm.quadratic_programming()
visual(test_X, test_y, w=svm.w)
y_pred = svm.predict(test_X)
acc = acc_estimator(test_y,y_pred)
print("Primal SVM acc:",acc)

# Dual SVM
svm = Dual_SVM(train_X, train_y)
svm.quadratic_programming()
visual(test_X, test_y, w=svm.w)
y_pred = svm.predict(test_X)
acc = acc_estimator(test_y,y_pred)
print("Dual SVM acc:",acc)


# Dual Kernel SVM
svm = Dual_Kernel_SVM(train_X, train_y, polynomial_kernel)
svm.quadratic_programming()
y_pred = svm.predict(test_X)
acc = acc_estimator2(test_y,y_pred)
print("Dual Kernel SVM acc:",acc)

# Soft_Margin_Dual_Kernel_SVM
svm = Soft_Margin_Dual_Kernel_SVM(train_X,train_y,gaussian_kernel,1.0)
svm.quadratic_programming()
y_pred = svm.predict(test_X)
acc = acc_estimator2(test_y,y_pred)
print("Soft Margin Dual Kernel SVM acc:",acc)