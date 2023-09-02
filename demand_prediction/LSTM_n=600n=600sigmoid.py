import torch
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import zipfile
import math
from shapely import geometry
import os
import torch
import sys
from sklearn.preprocessing import MinMaxScaler

# load some default Python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

supervised_data = pd.read_csv("LSTM_time_series.csv")
values = supervised_data.values[:,1:]

n_train_hours = 2000
n_step = 10
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

train_X, train_y = train[:, 0:2660], train[:,2660: 2926]
test_X, test_y = test[:, 0:2660], test[:, 2660: 2926]
train_X = train_X.reshape((train_X.shape[0], n_step, 266))
test_X = test_X.reshape((test_X.shape[0], n_step, 266))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM ,Dropout 
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint

n = 600

# 设计模型
model = Sequential()
model.add(LSTM(n, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model.add(LSTM(200,return_sequences=True))
#model.add(Dropout(0.01))
model.add(LSTM(n))
model.add(Dense(266))
model.compile(loss='mae', optimizer='adam')
# 拟合模型

filepath = "modn=600n=600.h5" #'my_model_3.h5'#保存的模型名字
#callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0),]
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)
callbacks_list = [checkpoint]
history = model.fit(train_X, train_y, epochs=500, validation_data=(test_X, test_y), verbose=2, shuffle=False,callbacks = callbacks_list)#batch_size=72, 
# 绘制损失趋势线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

import pickle
with open('modn=600n=600.pickle', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)

#from keras.models import load_model
#model = load_model('modn=600n=600.h5')
