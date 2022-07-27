# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:27:13 2022

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Dropout

class Two:
    def plot_trend(self, df, win_start, win_stop):
        '''
        Plot the trend line

        Parameters
        ----------
        df : DataFrame
            Series DataFrame.
        win_start : int
            Initial window size.
        win_stop : int
            Stop window size.

        Returns
        -------
        None.

        '''
        df = df[win_start:win_stop]
        plt.plot(df)
        plt.title(df.name)
        plt.show()

class Five:
    def train_test(self, X, win_size):
        '''
        Train test split for time series

        Parameters
        ----------
        X : DataFrame
            Selected features.
        win_size : int
            Past Days.

        Returns
        -------
        X_train : array
            X_train.
        y_train : array
            y_train.

        '''
        X_train = []
        y_train = []
        
        for i in range(win_size, len(X)):
            X_train.append(X[i-win_size:i])
            y_train.append(X[i])
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train
    
class Models:
    def dl_seq(self, input_shape, depth, layer, dropout, activation):
        '''
        Create sequential deep learning model

        Parameters
        ----------
        input_shape : int
            Input shape / X_train shape.
        depth : int
            Looping to create depth.
        layer : int
            Nodes for each respective depth.
        dropout : int
            Dropout for overfitting.
        activation : str
            Activation function for deep learning model.

        Returns
        -------
        model : function
            Deep learning model.

        '''
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        for i in range(depth):
            model.add(LSTM(layer, return_sequences=True))
            model.add(Dropout(dropout))
            
        model.add(LSTM(layer))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation))
        return model
    
    def plot_loss(self, hist, loss):
        '''
        Plotting for training and validation loss

        Parameters
        ----------
        hist : function
            Models fit.
        loss : str
            Activation for hist.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(hist.history[loss])
        plt.plot(hist.history['val_'+loss])
        plt.ylabel('Cases')
        plt.xlabel('Time')
        plt.legend(['Training '+loss, 'Validation '+loss])
        plt.show()
        
    def pl2_2line(self, y_test, y_pred):
        '''
        Plotting actual and predicted trend

        Parameters
        ----------
        y_test : array
            Actual values.
        y_pred : array
            Predicted values.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(y_test, color='red')
        plt.plot(y_pred, color='aqua')
        plt.legend(['Actual', 'Predicted'])
        plt.ylabel('Cases')
        plt.xlabel('Time')
        plt.show()
        
        
        
        
        
        
        
        
        
        
        