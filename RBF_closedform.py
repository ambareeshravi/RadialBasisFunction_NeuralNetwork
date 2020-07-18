#!/usr/bin/env python
# coding: utf-8
'''
Author: Ambareesh Ravi
Description: Radial Basis Function Neural Networks
Data: May 2, 2020
'''

import numpy as np
from sklearn.cluster import KMeans

# Utility functions
def calc_accuracy(y_act, y_pred):
    '''
    Calculates the accuracy of predictions

    Args:
        y_act = targets
        y_pred = predictions
    Returns:
        Accuracy as float
    '''
    if len(y_act.shape) > 1:
        y_act = np.argmax(y_act, axis = -1) # convert one hot to integer labels
        y_pred = np.argmax(y_pred, axis = -1) # convert one hot to integer labels
    return sum([1 if np.all(ya==yp) else 0 for ya,yp in zip(y_act, y_pred)]) / len(y_act) # calculates the accuracy

def shuffle_data(X, y):
    '''
    Shuffle the data randomly

    Args:
        X,y - np arrays to be shuffled
    Returns:
        X,y - shuffled np arrays
    '''
    indices = np.array(list(range(0,len(y))))
    np.random.shuffle(indices)
    return X[indices], y[indices]  

def train_test_split(X, y, test_size = 0.1, shuffle = True):
    '''
    Divide the data into train and test sets based on the given proportion

    Args:
        X - data as np array
        y - labels as np array
        test_size - test propotion as float
    '''
    if shuffle: X,y = shuffle_data(X,y) # shuffle before splitting
    split_point = int(len(y) * test_size)
    return X[split_point:], X[:split_point], y[split_point:], y[:split_point]

class RBF:
    def __init__(self, layer_units):
        '''
        Implements Radial Basis Function Neural Network
        '''
        assert len(layer_units) == 3, "RBF can only have 3 layers in total"
        self.input_units, self.hidden_units, self.output_units = layer_units
        self.predict = self.feed_forward
        
    def get_gaussian_params(self, X):
        '''
        Calculates the cluser centers for each hidden units
        
        Args:
            X - input as np array
        Returns:
            cluster centers as np array
        '''
        kmeans = KMeans(n_clusters=self.hidden_units)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        
        sigmas = list()
        for cluster_label in np.unique(kmeans.labels_):
            class_indices = np.argwhere(kmeans.labels_==cluster_label)
            sigmas.append(np.mean([self.euclidean(x, kmeans.cluster_centers_[cluster_label]) for x in X[class_indices]]))
        sigmas = np.array(sigmas)
        sigmas += 1e-10
        return centers, sigmas
    
    def euclidean(self, x, y):
        '''
        Calculates the Euclidean distance between 2 vectors
        '''
        return np.linalg.norm(x-y, axis = -1)
    
    def gaussian(self, centers, sigmas, inputs):
        '''
        Gaussian kernal for RBF

        Args:
            c - centre as  np array
            x - datapoint as  np array
        Returns:
            value of gaussian output as float
        '''
        gauss = lambda c,s,x: np.exp(- self.euclidean(c, x)**2 / (2 * s**2))
        return np.array([[gauss(c,s,x) for c,s in zip(centers, sigmas)] for x in inputs]).T
    
    def mean_squared_error(self, y_act, y_pred):
        '''
        Mean Square Error

        Args:
            y_act = targets
            y_pred = predictions
        Returns:
            error value as float
        '''
        return np.mean((y_act - y_pred)**2)
                               
    def feed_forward(self, inp):
        '''
        Calculates a forward pass for one training sample
        
        Network strictly contains 3 layers. So the calculation can be hard-coded instead of automation / loops

        Args:
            inp - input np array
        Returns:
            final output / forward pass information based on getInfo
        '''
        G = self.gaussian(self.centers, self.sigmas, inp)
        y_pred = np.dot(self.W.T, G)
        return y_pred
    
    def conv_predicted_labels(self, y_pred):
        '''
        Converts predictions into labels in the output format
        '''
        y_pred_copy = y_pred.copy()
        neg_indices = np.argwhere(y_pred_copy<=0)
        pos_indices = np.argwhere(y_pred_copy>0)
        y_pred_copy[pos_indices] = 1
        y_pred_copy[neg_indices] = -1
        return y_pred_copy
        
    def fit(self, X, y, centers = [], sigmas = [], val_size = 0.05, evaluate = False):
        
        if len(centers) == 0 or len(sigmas) == 0:
            new_centers, new_sigmas = self.get_gaussian_params(X)
            if len(centers) == 0: centers = new_centers
            if len(sigmas) == 0: sigmas = new_sigmas
        self.centers = centers
        self.sigmas = sigmas
        if len(self.sigmas) < 1: self.sigmas = np.array([self.sigmas]*len(centers))
        
        X,y = shuffle_data(X,y)
        
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size) # split into train and val sets
        else:
            X_train, y_train = X, y
        
        # Calcuate the output at the gaussian activation layer
        G = self.gaussian(self.centers, self.sigmas, X_train)
        
        # Least square regression [W = (GGt)^-1.G.Dt]
        self.W = np.dot(np.dot(np.linalg.pinv(np.dot(G, G.T)), G), y_train.T)
        
        if evaluate:
            # Train predictions
            y_train_pred = self.predict(X_train)
            train_loss = self.mean_squared_error(y_train, y_train_pred)
            train_accuracy = calc_accuracy(y_train, self.conv_predicted_labels(y_train_pred))

            # Val predictions
            val_loss, val_accuracy = 0.0, 0.0
            if val_size > 0:
                y_val_pred = self.predict(y_val)
                val_loss = self.mean_squared_error(y_val, y_val_pred)
                val_accuracy = calc_accuracy(y_val, self.conv_predicted_labels(y_val_pred))
            return train_loss, train_accuracy, val_loss, val_accuracy
