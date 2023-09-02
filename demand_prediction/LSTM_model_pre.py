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
from keras.models import load_model
# load some default Python modules
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM ,Dropout 
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint
n_step =10

def model_predict():
	supervised_data = pd.read_csv("LSTM_time_series.csv")
	values = supervised_data.values[:,1:]
	train = values[-1, :]

	#train_X, _ = train[:, 0:2660], train[:,2660:]
	train_X = train[ 266:]
	
	train_X = train_X.reshape((1, n_step, 266))	
	#date_time = pd.to_datetime(time_string)
	
	model = load_model('modn=600n=600.h5')

	pre_y = model.predict(train_X)
	pre_y = pre_y[0,:263]
	#print(pre_y.shape)
	output_dic = {j+1: 0 for j in range(263)}
	for i in range(len(pre_y)):
		output_dic[i+1]  = pre_y[i]

	return output_dic

