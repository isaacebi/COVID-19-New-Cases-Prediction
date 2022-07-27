# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:19:17 2022

@author: isaac
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

from main_module import Two, Five, Models

#%%

CSV_TEST_PATH = os.path.join(os.getcwd(), 
                             'datasets', 'cases_malaysia_test.csv')

CSV_TRAIN_PATH = os.path.join(os.getcwd(), 
                              'datasets', 'cases_malaysia_train.csv')

PICKLE_SAVE_PATH = os.path.join(os.getcwd(), 'models', 'MMS_ENCODER.pkl')

LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

BEST_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'best_model.h5')

#%% Step 1) Data Loading

# to make reproducible results 
tf.random.set_seed(7)

# read file
df_test = pd.read_csv(CSV_TEST_PATH, na_values=['?', ' '])
df_train = pd.read_csv(CSV_TRAIN_PATH, na_values=['?', ' '])

#%% Step 2) Data Inspection & Visualization

step = Two()

# inspecting the dataframe
df_test.info()
df_train.info()

# check for NaN's
df_test.isna().sum()
df_train.isna().sum()

# plot the trend line
step.plot_trend(df_train['cases_new'], win_start=0, win_stop=len(df_train))

#%% Step 3) Data Cleaning

# combine train and test dataframe
# train [:680], test=[681:] = 680+100 = 780
df = pd.concat((df_train, df_test))
df.shape # double check on the shape
df.isna().sum() # double check for the NaN's
df.interpolate(inplace=True) # filling missing data for time series
df.isna().sum() # double check for the NaN's

# after interpolate, some of the NaN's become .5 which in new_cases only
# int is allowed, therefore round down and change back to int
df['cases_new'] = df.cases_new.round(0).astype(int) 
df.info() # look back to the dataframe, so that cases_new -> int


#%% Step 4) Feature Selection

# copy before changing the dataframe
df2 = df.copy()
df2 = df2.cases_new.reset_index(drop=True) # select only cases_new

step.plot_trend(df2, 0, len(df2)) # plot cases_new trendline

#%% Step 5) Preprocessing

# used the past 30 days
win_size = 30
X_1 = df2[:680] # train datasets
X_2 = df2[680-win_size:] # test datasets

# double check on both shape
X_1.shape
X_2.shape

# scale X's into MinMax
mms = MinMaxScaler()
X_1 = mms.fit_transform(np.expand_dims(X_1, axis=-1)) # only fit train
X_2 = mms.transform(np.expand_dims(X_2, axis=-1)) # only transform test

# split the train and test into X_train, y_train, X_test, y_test
step = Five()
X_train, y_train = step.train_test(X_1, win_size)
X_test, y_test = step.train_test(X_2, win_size)

#%% Models

step = Models()
input_shape = np.shape(X_test)[1:] # input shape for robustness
depth = 1 # depth of models
layer = 64 # nodes of models
dropout = 0.2 # dropout to prevent overfitting
activation = 'relu' # relus is used cause the datasets has no negative

# create DL model
model_1 = step.dl_seq(input_shape, depth, layer, dropout, activation)
model_1.summary() # model summary
# models architecture
plot_model(model_1, show_shapes=(True), show_layer_names=(True))

# compiler
model_1.compile(optimizer='adam',
                loss='mse',
                metrics=['mean_absolute_percentage_error', 'mse'])

# callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)

# checkpoint and saving deep_learning model
mdc = ModelCheckpoint(BEST_MODEL_PATH,
                      monitor='mean_absolute_percentage_error',
                      save_best_only=True,
                      mode='min',
                      verbose=1)

# model training
hist = model_1.fit(X_train, y_train,
                   epochs=350, # training iteration
                   validation_data=(X_test, y_test),
                   callbacks=[tensorboard_callback, mdc])

#%%

print(hist.history.keys()) # get the keys

# training and validation
step.plot_loss(hist, 'loss')
step.plot_loss(hist, 'mse')
step.plot_loss(hist, 'mean_absolute_percentage_error')


#%%

# model prediction
y_pred = model_1.predict(X_test) 
step.pl2_2line(y_test, y_pred) # plotting scaled actual and predicted trendline

# inversed the scaled into actual values
actual_case = mms.inverse_transform(y_test)
predicted_case = mms.inverse_transform(y_pred)

# plotting the actual scale for actual and predicted trendline
step.pl2_2line(actual_case, predicted_case)

#%%

# metrics error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print('mae :', mae)
print('mse :', mse)
print('mape :', mape*100)


#%% pickle

# saving scaler encoder
with open(PICKLE_SAVE_PATH, 'wb') as file:
    pickle.dump(mms, file)









