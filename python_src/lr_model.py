import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    dataset_json = open("dataset/ff2ff_dataset_645.json", "r").read()
except:
    dataset_json = open("../dataset/ff2ff_dataset_645.json", "r").read()
dataset = json.loads(dataset_json)
alphabet = dataset[0]
sequence = dataset[1]
timing = dataset[2]
assert len(sequence) == len(timing)

seq_encoder = LabelEncoder()
seq_encoder.fit(alphabet)

num_sequence = len(sequence)

y = timing

def proc_sequence(seq):
    seq = np.array(seq)
    uniq, count = np.unique(seq, return_counts=True)
    uniq = seq_encoder.transform(uniq)

    ret = np.zeros((len(alphabet)))
    for i in range(len(uniq)):
        ret[uniq[i]] = count[i]
    return ret

X_lr = np.zeros((num_sequence, len(alphabet)))
for i in range(num_sequence):
    X_lr[i] = proc_sequence(sequence[i])

lr_model = LinearRegression()
_ = lr_model.fit(X_lr, y)

def predict_seq(seq):
    seq_arr = np.array([proc_sequence(seq)])
    pred = lr_model.predict(seq_arr)
    return pred
