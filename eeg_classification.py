import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import matplotlib

from sklearn.model_selection import train_test_split
from spectraltools import Spectral
from tensorflow import keras
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

import os

os.environ["NUMBA_DISABLE_CUDA"] = "1"


from sktime.datasets import load_from_tsfile
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame

import pandas as pd
import pickle


def spectral_model():
    mod = tf.keras.Sequential()
    mod.add(tf.keras.Input(shape=(x_train.shape[1])))
    mod.add(Spectral(units=200,
                     is_base_trainable=True,
                     is_diag_end_trainable=False,
                     is_diag_start_trainable=True,
                     activation='relu',
                     use_bias=True,
                     diag_end_initializer='Zeros',
                     diag_start_initializer='Ones',
                     diag_regularizer=tf.keras.regularizers.L1(l1=1e-1),
                     base_regularizer=tf.keras.regularizers.L1(l1=1e-4)))

    mod.add(Spectral(units=50,
                     is_base_trainable=True,
                     is_diag_end_trainable=False,
                     is_diag_start_trainable=True,
                     activation='tanh',
                     use_bias=True,
                     diag_end_initializer='Zeros',
                     diag_start_initializer='Ones'))
    mod.add(Spectral(units=50,
                     is_base_trainable=True,
                     is_diag_end_trainable=False,
                     is_diag_start_trainable=True,
                     activation='tanh',
                     use_bias=True,
                     diag_end_initializer='Zeros',
                     diag_start_initializer='Ones'))
    mod.add(Spectral(1,
                     is_base_trainable=True,
                     is_diag_end_trainable=False,
                     is_diag_start_trainable=True,
                     activation='linear',
                     use_bias=True,
                     diag_end_initializer='Zeros',
                     diag_start_initializer='Ones'))
    return mod

#%%


with open('tsfresh_train.pkl', 'rb') as f:
    df = pickle.load(f)
with open('tsfresh_test.pkl', 'rb') as f:
    df_test = pickle.load(f)

df_0 = df[df['target'] == '0']
df_1 = df[df['target'] == '1']

n_min = len(df_0)

df_1_sampled = df_1.sample(n=n_min, random_state=42)

df_balanced = pd.concat([df_0, df_1_sampled]).sample(frac=1, random_state=42)  # shuffle finale

y = df_balanced.target.values
y_train = np.zeros(len(y))
y_train[y == '1'] = 1

y_test = df_test.target.values
y_test[y_test == '1'] = 1
y_test[y_test == '0'] = 0

f = df_balanced.fillna(-1).drop('target', axis=1)
x_train = f.values
x_test = df_test.fillna(-1).drop('target', axis=1).values


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)




rel = []
for iter in range(2):
    mod = spectral_model()
    mod.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics='accuracy',
    )


    hist = mod.fit(x_train_scaled, y_train,
                   batch_size=100,
                   epochs= 500,
                   validation_split=0.1,
                   verbose=1)

    eig_start = Spectral.return_diag(mod.layers[0])
    base_start = mod.layers[0].base
    norma = tf.linalg.norm(base_start, ord=2, axis=1)
    eig_norm = (eig_start * norma)
    eig_norm = eig_norm.numpy()

    n_rel = np.sum(np.abs(eig_norm) > 0.001)

    rel.append(np.argsort(np.abs(eig_norm))[-n_rel:])

    pred = tf.sigmoid(mod(x_test_scaled)).numpy()[:, 0]
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    print('accuracy test: ' + str((y_test == pred).mean()))


plt.hist(np.abs(eig_norm), 40)
plt.yscale('log')
plt.show()



# print name of relevant features

rc = rel[0]
for i in range(1, len(rel)):
    rc = np.concatenate([rc, rel[i]])

sr = set(rc)

count = np.zeros(len(sr))
i = -1
for s in sr:
    i += 1
    for r in rel:
        if s in r:
            count[i] += 1

rel_feat_index = np.array(list(sr))[count >= len(rel)-1]

for i in rel_feat_index:
    print(df.columns[i])

