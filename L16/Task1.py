import pandas as pd
import numpy as np
import warnings
from utils import *
warnings.filterwarnings('ignore')
np.random.seed(49)

path = r"E:\课内学习\模式识别（Part Two）\hw\L16编程作业实验数据集\实验数据集\Iris数据集\iris.csv"
data = pd.read_csv(path, index_col=0)
kinds = data['Species'].unique()

cnt = 0
trains = []
tests = []

for kind in kinds:
    train, test = split_dataset(data, kind, 30)
    trains.append(train)
    tests.append(test)
train = pd.concat(trains,axis=0).reset_index(drop=True)
test = pd.concat(tests, axis=0).reset_index(drop=True)
test_copy = test.copy(deep=True) # prepare for softmax classify

# OvO classify
for k in range(len(kinds)):
    for j in range(k+1, len(kinds)):
        cnt = cnt + 1
        poskind = kinds[k]
        negkind = kinds[j]
        partial_data = resplit_dataset(train,poskind,negkind)
        clf = PLA_Classifer(partial_data, poskind, negkind)
        clf.fit(epoch_num=20, lr=0.5)
        test[f'pred{cnt}'] = clf.predict(test)

test = pd.DataFrame(test)
vote_out(test)
acc = acc_estimate(test)
print(test)
print("Accuracy on test dataset by OvO method:",acc)
print("-------------------------")

"""
Record:
1.需要一开始就分层划分出训练集与测试集，90：60
2.使用多条件布尔索引时：(x['a'] == m) | (x['a'] == n)，不要偷懒直接用x['a'] == (m|n)
3.第一次打乱顺序是在分层划分训练集与测试集时，合并子训练集后需要再打乱一次，这样感知机再在训练时面对的标签不是111111111，-1-1-1-1-1-1-1，而是交替出现，这样才有利于训练
4.设置早退逻辑时,if_wrong的逻辑判断位置不要放错了
5.由于代码原因，test的列是不断增多的，因此test划分特征向量与标签时，不要用-1索引，用正向索引
"""
print("Below are records of softmax method")
# Softmax Classify
train = train
test = test_copy

softmax_clf = SoftmaxClassifier(0,1)
losses = softmax_clf.fit(train, epoch_num=20, batch_size=20, lr=0.1)
pred = softmax_clf.predict(test)
test['softmax_pred'] = pred
acc = acc_estimate(test)
print("test dataset with prediction:")
print(test)
print("Accuracy on test dataset by softmax method:",acc)
print("loss decrease by batch:",losses)

"""
Record:
1.在序列遍历时，发生两类错误：
 1.1 for i in len(y)  len(y)是一个整数，不能遍历 -> for i in range(len(y))
 1.2 for i in y       i不是真实的顺序           -> for i in range(len(y))
2.参数更新板块：
 先取w = W[k]，然后再对w进行梯度更新，这样是错误的，因为更新的w并不会使W[k]更新，w是W[k]的深度拷贝，将w直接用W[k]表示即可解决问题
3.在多次实验中，发现了softmax对超参数的设置更敏感，且最终在测试集上的分类准确率并不总优于OvO法
4.softmax对参数矩阵初始化方式敏感
"""