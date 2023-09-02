import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path of csv')
args = parser.parse_args()
filePath = args.input

df_train = pd.read_csv(filePath,  parse_dates=["tpep_pickup_datetime"])
#df_train_6 = pd.read_csv('06.csv',  parse_dates=["tpep_pickup_datetime"])#nrows = 2_000_000,
#df_train_7 = pd.read_csv('07.csv',  parse_dates=["tpep_pickup_datetime"])
#df_train_8 = pd.read_csv('08.csv',  parse_dates=["tpep_pickup_datetime"])
#df_train = pd.concat([df_train_6, df_train_7, df_train_8],axis=0,ignore_index=True)
df_train["tpep_dropoff_datetime"] =pd.to_datetime(df_train["tpep_dropoff_datetime"])
df_train["tpep_pickup_datetime"] =pd.to_datetime(df_train["tpep_pickup_datetime"])

print("df_train:",df_train)

def taxi_data_preprocessing(df_train,freq):
    
    df_train["tpep_pickup_datetime"] =pd.to_datetime(df_train["tpep_pickup_datetime"])#转换数据类型
    datetime_range = pd.date_range(start=df_train["tpep_pickup_datetime"].min(), end=df_train["tpep_pickup_datetime"].max(), periods=None, freq = freq) #freq = '1H'

    df_train_Loc = pd.DataFrame(index=df_train["tpep_pickup_datetime"])
    df_train_Loc = df_train_Loc.resample(freq).sum() #freq= "H"
    for i in range(1,df_train["PULocationID"].max()+1):
        df_train_Loc_temp = df_train.loc[df_train["PULocationID"] == i]#筛选地区
        ts = pd.DataFrame(np.ones([len(df_train_Loc_temp["tpep_pickup_datetime"]),1]),
                        index=df_train_Loc_temp["tpep_pickup_datetime"]
                        ,columns = [i])
        df_train_Loc = pd.merge(df_train_Loc,ts.resample(freq).sum(),on = "tpep_pickup_datetime", how ="outer")

    df_train_Loc_Time = df_train_Loc.fillna(0)

    
    return df_train_Loc_Time


df_train_Loc_Time = taxi_data_preprocessing(df_train, freq = "H" )
print("df_train_Loc_Time:",df_train_Loc_Time)
df_train_Loc_Time.to_csv("data.csv")

#########################################2###############################################3
import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
import tensorflow as tf
from tensorflow import keras
import zipfile
import math

df_train_Loc_Time =  pd.read_csv('data.csv')
df_train_Loc_Time['tpep_pickup_datetime'] =pd.to_datetime(df_train_Loc_Time["tpep_pickup_datetime"])
df_train_Loc_Time = df_train_Loc_Time.drop(['264','265'],axis = 1)

x = df_train_Loc_Time.copy()

#df_train_Loc_Time["year"] = x['tpep_pickup_datetime'].map(lambda x:x.year)
df_train_Loc_Time["month"] = x['tpep_pickup_datetime'].map(lambda x:x.month)
df_train_Loc_Time["day"] = x['tpep_pickup_datetime'].map(lambda x:x.day)
df_train_Loc_Time["hour"] = x['tpep_pickup_datetime'].map(lambda x:x.hour)
#df_train_Loc_Time.drop("tpep_dropoff_datetime",axis = 1)
print(df_train_Loc_Time)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

LSTM_time_series = series_to_supervised(np.array(df_train_Loc_Time.values[:,1:]),10)
print("LSTM_time_series:",LSTM_time_series)
LSTM_time_series.to_csv("LSTM_time_series.csv")

