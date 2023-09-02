import torch
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import zipfile
import math
import geopandas
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

#geo data
data = geopandas.read_file("taxi_zones.json")

data['centroid_column'] = data.centroid
coord_list = [(x,y) for x,y in zip(data['centroid_column'].x , data['centroid_column'].y)]
coord_list = np.array(coord_list)
from sklearn.preprocessing import MinMaxScaler
#coord_list_min_max = MinMaxScaler().fit_transform(coord_list)

scaler = MinMaxScaler()
scaler.fit(coord_list)
coord_list_min_max = scaler.transform(coord_list)
#scaler.inverse_transform() 
#coord_list_min_max
print("—————————js file loaded———————————")

df_train_6 =  pd.read_csv('06.csv',  parse_dates=["tpep_pickup_datetime"])#nrows = 2_000_000,
df_train_7 =  pd.read_csv('07.csv',  parse_dates=["tpep_pickup_datetime"])
df_train_8 =  pd.read_csv('08.csv',  parse_dates=["tpep_pickup_datetime"])
df_train = pd.concat([df_train_6, df_train_7, df_train_8],axis=0,ignore_index=True)




df_train["tpep_dropoff_datetime"] =pd.to_datetime(df_train["tpep_dropoff_datetime"])

x = df_train.copy()


train_X = pd.DataFrame()
#train_X["year"] = x['tpep_dropoff_datetime'].map(lambda x:x.year)
train_X["month"] = x['tpep_dropoff_datetime'].map(lambda x:x.month)
train_X["day"] = x['tpep_dropoff_datetime'].map(lambda x:x.day)
train_X["hour"] = x['tpep_dropoff_datetime'].map(lambda x:x.hour)
train_X["minute"] = x['tpep_dropoff_datetime'].map(lambda x:x.minute)
train_X["second"] = x['tpep_dropoff_datetime'].map(lambda x:x.second)

#train_X = train_X.values.reshape(train_X.shape[0],train_X.shape[1])
#train_X.to_csv("total_data.csv")
#y.to_csv("y.csv")
##########################################################
#train_X = pd.read_csv('total_data.csv')
#print(train_X.head())
#y =  pd.read_csv('y.csv')
#print(y.head())
########################################################
print("—————————csv loading finished———————————")

#y = pd.DataFrame()

#y['DOLocationID'] = df_train['DOLocationID']
coord_list = np.r_[coord_list,[[np.nan,np.nan]]]
coord_list = np.r_[coord_list,[[np.nan,np.nan]]]
train_y = coord_list[y['DOLocationID'].values-1]



a = np.concatenate((train_X,train_y),axis=1)
train = a[~np.isnan(a).any(axis=1)]

#train = a[:n_train_samples, :]
#test = a[n_train_samples:, :]

train_X,train_y = train[:,1:6],train[:,6  :8]

train_X = train_X.reshape(train_X.shape[0],1,train_X.shape[1])
train_y[:,0] = -train_y[:,0]
print("train_X.shape,train_y.shape:", train_X.shape,train_y.shape)

"""
A Mixture Density Layer for Keras
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer
Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
for a starting point for this code.
Provided under MIT License
"""
#from .version import __version__
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import layers
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return keras.backend.elu(x) + 1 + keras.backend.epsilon()


class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.
    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
            self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                         name='mdn_outputs')
        return mdn_out

    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # classmethod
    # def from_config(cls, config):
    #     return cls(**config)


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func


def get_mixture_sampling_fun(output_dim, num_mixes):
    """Construct a TensorFlor sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    def sampling_func(y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        # Todo: temperature adjustment for sampling function.
        return samp

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func


def get_mixture_mse_accuracy(output_dim, num_mixes):
    """Construct an MSE accuracy function for the MDN layer
    that takes one sample and compares to the true value."""
    # Construct a loss function with the right number of mixtures and outputs
    def mse_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        samp = mixture.sample()
        mse = tf.reduce_mean(tf.square(samp - y_true), axis=-1)
        # Todo: temperature adjustment for sampling functon.
        return mse

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return mse_func


def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    mus = params[:num_mixes * output_dim]
    sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    """Samples from a categorical model PDF.
    Arguments:
    dist -- the parameters of the categorical model
    Returns:
    One sample from the categorical model, or -1 if sampling fails.
    """
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info('Error sampling categorical model.')
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)
    Returns:
    One sample from the the mixture model.
    """
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim:(m + 1) * output_dim]
    sig_vector = sigs[m * output_dim:(m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample

from tensorflow import keras
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from keras.utils import plot_model
OUTPUT_DIMS = 2
N_MIXES =40 #用来拟合分布的个数
N_HIDDEN_1 = 100
N_HIDDEN_2 = 100


model = Sequential()
model.add(keras.layers.Dense(N_HIDDEN_1, batch_input_shape=(None, 5), activation='sigmoid'))
model.add(keras.layers.Dense(N_HIDDEN_2, activation='sigmoid'))
#model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(200))
model.add(MDN(OUTPUT_DIMS, N_MIXES))
model.compile(loss=get_mixture_loss_func(OUTPUT_DIMS,N_MIXES), optimizer=keras.optimizers.Adam())

model.summary()
plot_model(model, to_file='model.png')

print("—————————模型搭建完成———————————")
from keras.callbacks import EarlyStopping, ModelCheckpoint


filepath = "Model_M=40_n1=100_n2=100sigmoid.h5" #'my_model_3.h5'#保存的模型名字
#callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0),]
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
callbacks_list = [checkpoint]

history = model.fit(x=train_X, y=train_y, epochs=100,batch_size=100000,  validation_split=0.30, callbacks= callbacks_list)#batch_size=72, 



###################################储存model###############################################

#model.save(filepath)

###################################储存history###############################################

import pickle
with open('Model_M=40_n1=100_n2=100sigmoid.pickle', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)

from matplotlib import pyplot


pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.savefig("history.png")
pyplot.legend()
pyplot.show()
plt.figure(0)
pyplot.figure(0)
#model.load_weights("vi_128_lrdefault.hdf5")




train_X = train_X.reshape(train_X.shape[0],train_X.shape[2])
y_input = train_X[200:500]

################

print("y_input:",y_input)
x_test = model.predict(y_input)
print("x_test:",x_test)


y_samples = np.apply_along_axis(sample_from_output, 1, x_test, OUTPUT_DIMS, N_MIXES, temp=1.0, sigma_temp=1.0)
#print("y_samples:",y_samples)
import matplotlib.pyplot as plt
y_samples = y_samples.reshape(y_samples.shape[0],y_samples.shape[2])
image = plt.scatter(y_samples[:,0],y_samples[:,1])
#image.show()
plt.savefig("Model_M=40_n1=100_n2=100sigmoid.png")
plt.show
plt.close(0)
plt.figure(0)

#y_samples = train_y
#import matplotlib.pyplot as plt
#image = plt.scatter(y_samples[:,0],y_samples[:,1])

#plt.savefig("scatter_plot_true.png")
#plt.show
#plt.close(0)
#plt.figure(0)

import matplotlib.pyplot as pl
import scipy.stats as st

data = train_y[1:100000]

x = data[:, 0]

y = data[:, 1]

xmin, xmax = data[:, 0].max(), data[:, 0].min()

ymin, ymax = data[:, 1].max(), data[:, 1].min()

# Peform the kernel density estimate

xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values = np.vstack([x, y])

kernel = st.gaussian_kde(values)

f = np.reshape(kernel(positions).T, xx.shape)

fig = pl.figure()

ax = fig.gca()

ax.set_xlim(xmin, xmax)

ax.set_ylim(ymin, ymax)

# Contourf plot

cfset = ax.contourf(xx, yy, f, cmap='Blues')

## Or kernel density estimate plot instead of the contourf plot

#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

# Contour plot

cset = ax.contour(xx, yy, f, colors='k')

# Label plot

ax.clabel(cset, inline=1, fontsize=10)

ax.set_xlabel('Y1')

ax.set_ylabel('Y0')

pl.show()