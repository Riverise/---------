import pandas as pd
import numpy as np
from utils import load_mnist, SoftmaxClassifier
np.random.seed(42)

X_train, y_train = load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\mnist', kind='train')
X_test, y_test = load_mnist(r'E:\CODE\JupyterWork\DIAN2024春招算法\mnist', kind='t10k')

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
clf = SoftmaxClassifier(0,0.01)
losses = clf.fit(train_data,10,256,1)
y_pred = clf.predict(test_data)

y_true = test_data.iloc[:,-1]
acc = sum(y_pred == y_true) / len(y_pred)
print("Accuracy on test dataset:",acc)

# sample
print("Below are ten samples got randomly")
sample_idx = np.random.choice(range(len(y_true)), size=10, replace=False)
print("True labels:", y_true[sample_idx])
print("Predicted labels:", y_pred[sample_idx])

"""
Record:
1.softmax时很容易因为np.exp()操作而上溢，解决办法就是对输入序列的所有元素加上一个偏置（z = x - max(x)）
2.同样地，因为softmax输入序列的极差较大，会导致输出的概率分布存在下溢情况，接近于0，由于计算机浮点数不精确存储，可能会直接存储为0，
 这就会导致在后续计算交叉熵时损失为无穷大inf（故没绘制损失曲线），但是可以看到训练后的分类器在测试集上的表现还是不错的
3.由于SoftmaxClassifier类与第一题共用，在此基础添加在每一轮训练中计算训练精度与测试精度可能会对第一题造成影响，故本题未完成题目的所有细节。
4.未对样本特征向量添加偏置项
"""