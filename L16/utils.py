import numpy as np
import pandas as pd
import math
import os
import gzip
from collections import Counter
from tqdm import tqdm

def split_dataset(dataset: pd.DataFrame, kind:str, train_num:int) -> pd.DataFrame:
    # train_test_split
    mask = dataset['Species'] == kind
    data_wanted = dataset[mask]
    shuffle_data = data_wanted.sample(frac=1)
    train = shuffle_data[:train_num]
    test = shuffle_data[train_num:]
    return train, test

def resplit_dataset(dataset: pd.DataFrame, poskind:str, negkind:str) -> pd.DataFrame:
    # Get partial dataset according OvO method"""
    return dataset[(dataset['Species'] == poskind)|(dataset['Species'] == negkind)]

def acc_estimate(df:pd.DataFrame) -> float:
    correct = df['Species'] == df.iloc[:,-1]
    return sum(correct) / len(correct)

def vote_out(df:pd.DataFrame) -> None:
    def vote(row):
        preds = [row['pred1'], row['pred2'], row['pred3']]
        counter = Counter(preds)
        final_pred = counter.most_common(1)[0][0]
        return final_pred
    df['final_pred'] = df.apply(vote,axis=1)

def cross_entropy(y:np.ndarray,y_pred:np.ndarray) -> float:
    assert len(y) == len(y_pred), "Dimension mismatch!"
    losslist = []
    for i in range(len(y)):
        idx = y[i]
        p = y_pred[i][idx]
        losslist.append(-1 * np.log(p)) # in Task2 p can be very close to zero, so the cross entropy can be inf
        loss = sum(losslist)
    return loss

def update(W:np.ndarray, X:np.ndarray, y:np.ndarray, y_pred:np.ndarray, lr:float):
    # X:(batch_size, feature_num)
    # W:(cate_num, feature_num)
    # batch_y_pred:(batch_size, cate_num)
    for i in range(len(X)): # the sample with index of i
        x = X[i]
        idx = y[i] # the true label
        p = y_pred[i][idx]
        for k in range(len(W)):
            if k == idx: 
                grad = (p-1)*x
            else:
                n = y_pred[i][k]
                grad = n*x
            W[k] = W[k] - lr * grad
    return W

class PLA_Classifer:
    def __init__(self, dataset: pd.DataFrame, poskind:str, negkind:str) -> None:
        self.poskind = poskind
        self.negkind = negkind
        self.dataset = dataset.sample(frac=1).reset_index(drop=True)

    def fit(self, epoch_num:int=10, lr:float=0.1) -> np.ndarray:
        train_X = self.dataset.iloc[:,:-1]
        train_y = self.dataset.iloc[:, -1]

        mask = train_y == self.poskind
        train_y[mask] = 1
        train_y[~mask] = -1

        # augmentation 
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        X = np.insert(train_X ,0 ,1, axis=1)
        w0 = np.zeros_like(X[0])
        for _ in range(epoch_num):
            if_wrong = False
            for i in range(len(X)):
                if (w0 @ X[i]) * train_y[i] <= 0:
                    w0 += train_y[i] * X[i] * lr
                    if_wrong = True
            if ~if_wrong:
                break
        self.w = w0

    def predict(self, data):
        X = data.iloc[:,:4]
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        result = np.sign(X @ self.w)
        mask = result == 1
        y_pred = pd.Series(list('_' * len(data)))
        y_pred[mask] = self.poskind
        y_pred[~mask] = self.negkind
        return y_pred
    
class SoftmaxClassifier:
    def __init__(self, loc, scale) -> None:
        # miu and sigma parameters for normal initialization
        self.loc = loc
        self.sacle = scale             

    def fit(self, train_data:pd.DataFrame, epoch_num:int=20, batch_size:int=20, lr:float=0.1):
        self.categories = train_data['Species'].unique()
        cate_num = len(self.categories)
        # label_encode
        encode = {}
        for id, category in enumerate(self.categories):
            encode[category] = id
        train_data['Species'] = train_data['Species'].map(encode)
        # shuffle
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        # split feature and label
        X = train_data.iloc[:,:-1]
        y = train_data.iloc[:,-1]
        # augmentation
        X = np.array(X)
        y = np.array(y)
        X = np.insert(X, 0, 1,axis=1)
        self.W = np.random.normal(self.loc, self.sacle, (cate_num, X.shape[1]))

        losses = []
        for _ in range(epoch_num):
            for batch_id in tqdm(range(math.ceil(len(X)/batch_size))):
                if (batch_id + 1) * batch_size > len(X):
                    batch_X = X[batch_id*batch_size: ]
                    batch_y = y[batch_id*batch_size: ]
                else:
                    batch_X = X[batch_id*batch_size : (batch_id+1)*batch_size]
                    batch_y = y[batch_id*batch_size : (batch_id+1)*batch_size]
                
                # X:(batch_size, feature_num)
                # W:(cate_num, feature_num)
                S = batch_X @ self.W.T # S:(batch_size, cate_num)
                def softmax(row):
                    row = row - np.max(row) # Protection from Exp Overflow
                    row = np.exp(row)
                    return row / sum(row)
                  
                batch_y_pred = np.apply_along_axis(softmax,1,S) # batch_y_pred:(batch_size, cate_num)
                loss = cross_entropy(batch_y, batch_y_pred)
                self.W = update(self.W, batch_X, batch_y, batch_y_pred, lr)
                # break # for debug
                losses.append(loss) # Record loss by batch           
        return losses

    def predict(self, data) -> pd.Series:
        # X:(batch_size, feature_num)
        # W:(cate_num, feature_num)
        X = data.iloc[:,:-1]
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        decode = {}
        for id, category in enumerate(self.categories):
            decode[id] = category
        S = X @ self.W.T
        y_pred = np.argmax(S,axis=1)
        y_pred = pd.Series(y_pred)
        return y_pred.map(decode)


# -------------------------------------------------------------------
# Not used this time. 
def data_iter(batch_size, X_train, y_train):
    # Batch data for training
    indices = list(range(len(X_train)))
    random.shuffle(indices)
    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i: min(i + batch_size, len(X_train))]
        yield X_train[batch_indices], y_train[batch_indices]
# -------------------------------------------------------------------

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    
    return pd.DataFrame(images), pd.DataFrame(labels,columns=['Species'])