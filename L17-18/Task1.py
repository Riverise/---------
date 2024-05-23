import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from typing import List
warnings.filterwarnings('ignore')
np.random.seed(49)

def split_dataset(dataset: pd.DataFrame, kind:str, train_num:int) -> pd.DataFrame:
    # train_test_split
    mask = dataset['Species'] == kind
    data_wanted = dataset[mask]
    shuffle_data = data_wanted.sample(frac=1)
    train = shuffle_data[:train_num]
    test = shuffle_data[train_num:]
    return train, test

def data_iter(batch_size, X_train, y_train):
    # Batch data for training
    indices = list(range(len(X_train)))
    np.random.shuffle(indices)
    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i: min(i + batch_size, len(X_train))]
        yield X_train[batch_indices], y_train[batch_indices]

def acc_estimator(y_true:torch.Tensor, y_pred:torch.Tensor)->float:
    return np.mean(np.array(y_true == y_pred))

path = r"E:\课内学习\模式识别（Part Two）\hw\L16编程作业实验数据集\实验数据集\Iris数据集\iris.csv"
data = pd.read_csv(path, index_col=0)
kinds = data['Species'].unique()

categories = data['Species'].unique()
cate_num = len(categories)
# label_encode
encode = {}
decode = {}
for id, category in enumerate(categories):
    encode[category] = id
    decode[id] = category

trains = []
tests = []

for kind in kinds:
    train, test = split_dataset(data, kind, 30)
    trains.append(train)
    tests.append(test)
train = pd.concat(trains,axis=0).reset_index(drop=True)
test = pd.concat(tests, axis=0).reset_index(drop=True)

train['Species'] = train['Species'].map(encode)
test['Species'] = test['Species'].map(encode)

train_X = torch.tensor(train.iloc[:,:-1].values,dtype=torch.float32 )
train_y = torch.tensor(train.iloc[:,-1].values,dtype=torch.long)
test_X = torch.tensor(test.iloc[:,:-1].values,dtype=torch.float32)
test_y = torch.tensor(test.iloc[:,-1].values,dtype=torch.long)

class Net(nn.Module):
    def __init__(self, input_size, hidden_dims:List, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_size))
        self.mlp = nn.Sequential(*layers)
        self.config = str(hidden_dims)
    
    def forward(self, x):
        return self.mlp(x)

def train(num_epochs, batch_size, net, optimizer, criterion):
    net.train()
    for _ in range(num_epochs):
        for (batch_X, batch_y) in data_iter(batch_size, train_X, train_y):
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

def evaluate(net, test_X, test_y):
    net.eval()
    with torch.no_grad():
        outputs = net(test_X)
        _, y_pred = torch.max(outputs,1)
        return acc_estimator(test_y, y_pred)


configs = [
    {'input_size': 4, 'hidden_dims': [128], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [128, 128], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [128, 128, 128], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [128, 256], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [256, 256], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [256, 384], 'output_size': 3},
    {'input_size': 4, 'hidden_dims': [384, 384], 'output_size': 3},
]

config_infos = []
accuracies = []

epoch = 50
lr = 0.1
bs = 5

for config in configs:
    net = Net(**config)
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=lr)
    train(num_epochs=epoch, batch_size=bs, net=net, optimizer=sgd, criterion=criterion)
    accuracy = evaluate(net, test_X, test_y)

    # store relevant data
    config_infos.append(str(config['hidden_dims']))
    accuracies.append(accuracy)

df = pd.DataFrame({ 'accuracy': accuracies}, index=config_infos)
print(df)

"""
实验结论：
较低epoch时神经元越多，层数越深，看上去效果更差，实际上是因为参数量更大，输入层梯度较小，网络权重还没收敛就已停止迭代；
随着调高epoch与learning_rate，参数量更大的模型效果也逐步提升，但是简单的128个一层神经元对于处理这个分类问题已绰绰有余，
这个问题还没复杂到需要用到非常深的网络去学习其潜在的特征
"""