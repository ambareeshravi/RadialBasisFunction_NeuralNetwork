#!/usr/bin/env python
# coding: utf-8

'''
Author: Ambareesh Ravi
Description: Implementation of RBF NN from scratch with backprop
Data: May 4, 2020
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
    def __init__(self, layer_units, output_activation = "tanh", useBias = True, lrDecay = 0.75):
        '''
        Implements Radial Basis Function Neural Network
        '''
        assert len(layer_units) == 3, "RBF can only have 3 layers in total"
        self.input_units, self.hidden_units, self.output_units = layer_units
        self.useBias = useBias
        self.output_activation = output_activation
        
        self.network = self.build_network()
        self.lrDecay = lrDecay
        
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
        sigmas = [(np.linalg.norm(X-c)**2 / len(X)) for c in centers] # change
        return centers, sigmas
    
    def euclidean(self, x, y):
        '''
        Calculates the Euclidean distance between 2 vectors
        '''
        return np.linalg.norm(x-y, axis = -1)
        
    # Define activation functions and derivatives   
    def least_square_regression(self, X, y):
        '''
        Finding weights without backpropogation
        '''
        W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        return W
    
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
        return np.array([[gauss(c,s,x) for c,s in zip(centers, sigmas)] for x in inputs])
    
    def tanh(self, x):
        '''
        Calculates the tanh of a value or an array

        Args:
            x - float/np array
        Returns:
            float / array with tanh(s)
        '''
        return np.tanh(x)
    
    def d_tanh(self, x):
        return 1 - self.tanh(x)**2
    
    def identity(self, x):
        return x
    
    def d_identity(self, x):
        return 1

    # Define loss functions and derivatives
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

    def d_mse(self, y_act, y_pred):
        '''
        Calculates the derivative of squared error

        Args:
            y_act = targets
            y_pred = predictions
        Returns:
            derivative value as float
        '''
        return (y_pred - y_act) * (2 / len(y_act))
        
    def build_network(self):
        '''
        Builds an RBF network based on the number of input, hidden and output layers
        
        Returns:
            list of layer information as dict
            
        w_ji: weight between i_th neuron in layer l and j_th neuron in layer l+1
        '''
        # hidden -> output
        self.W = np.random.randn(self.hidden_units, self.output_units) # / 10.
        if self.useBias: self.b = np.random.randn(self.output_units) # / 10.
        
        if self.output_activation in ["tanh"]:
            self.output_activation = self.tanh
            self.d_output_activation = self.d_tanh
        if self.output_activation in ["identity"]:
            self.output_activation = self.identity
            self.d_output_activation = self.d_identity
            
        self.loss_function = self.mean_squared_error
        self.d_loss = self.d_mse
               
    def feed_forward(self, inp, returnInfo = False):
        '''
        Calculates a forward pass for one training sample
        
        Network strictly contains 3 layers. So the calculation can be hard-coded instead of automation / loops

        Args:
            inp - input np array
        Returns:
            final output / forward pass information based on getInfo
        '''
        a1 = self.gaussian(self.centers, self.sigmas, inp)
        z2 = np.dot(a1, self.W)
        a2 = self.output_activation(z2)
        if self.useBias: a2 += self.b
        if returnInfo: return {"a1": a1, "z2": z2, "a2": a2}
        return a2
    
    def back_propogation(self, y_act, y_pred, a1, z2):
        '''
        Calculates the backward propogation

        Args:
            foward_pass_info - list of outputs and activations at each layer
            y_act - targets
            x - input
        '''
        residue = self.d_loss(y_act.T, y_pred.T)
        dw = residue * np.multiply(self.d_output_activation(z2), a1)
        db = residue
        return dw, db
    
    def conv_predicted_labels(self, y_pred):
    	'''
    	Roudning off labels for tanh
    	'''
        neg_indices = np.argwhere(y_pred<=0)
        pos_indices = np.argwhere(y_pred>0)
        y_pred[pos_indices] = 1
        y_pred[neg_indices] = -1
        return y_pred
        
    def fit(self, X, y, lr, epochs, gauss_params = None, batch_size = 32, val_size = 0.05):
    	'''
    	Fits RBF NN on given data

    	Args:
    		X - data as <np.array>
    		y - labels as <np.array>
    		lr - learning rate as <float>
    		epochs - epochs as <int>
    		gauss_params -  a <dict> containing "centers" and "sigmas" as keys
    		batch_size - <int>
    		val_size - validation set size as <float>

    	Returns:
    		-

    	Exception:
    		-

    	Example:
	    	gauss_params = dict()
			gauss_params["centers"] = X_train[:150]
			gauss_params["sigmas"] = np.ones(len(X_train))[:150]

			rbf = RBF([2,len(X_train),1], output_activation='tanh')
			rbf.fit(X, y, lr = 0.08, gauss_params = None, epochs = 100, batch_size=32, val_size=0.01)
    	'''
        if gauss_params == None:
            self.centers, self.sigmas = self.get_gaussian_params(X)
        else:
            self.centers = gauss_params["centers"]
            self.sigmas = gauss_params["sigmas"]
        
        X,y = shuffle_data(X,y) # shuffle data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size) # split into train and val sets
        
        data_len = len(y_train)
        for epoch in range(1, epochs+1):
            X_train, y_train = shuffle_data(X_train, y_train) # Shuffle data at the start of every epoch
            if self.lrDecay > 0 and epoch % 10 == 0: lr *= self.lrDecay # Implementation for learning rate decay for better convergence
            train_losses, train_accuracies = list(), list()
            for str_idx in range(0, data_len, batch_size):
                # get batch data
                end_idx = str_idx + batch_size 
                if end_idx > batch_size: continue
                X_batch = X_train[str_idx:end_idx]
                y_batch = y_train[str_idx:end_idx]

                batch_delta_w = list()
                if self.useBias: batch_delta_b = list()

                # do for each sample
                for xi, yi in zip(X_batch, y_batch):
                    if xi.ndim < 2: xi = xi.reshape(1,-1) # check shape
                    if yi.ndim < 2: yi = yi.reshape(1,-1) # check shape
                    # forward pass
                    fp_info = self.feed_forward(xi, True)
                    # backpropagate
                    delta_w, delta_b = self.back_propogation(yi, fp_info["a2"], fp_info["a1"], fp_info["z2"])
                    batch_delta_w.append(delta_w) # save changes in weights
                    if self.useBias: batch_delta_b.append(delta_b)
                    
                # update changes in weights and biases
                batch_delta_w = np.mean(np.array(batch_delta_w), axis=0).T
                self.W -= (lr * batch_delta_w)
                if self.useBias:
                    batch_delta_b = np.mean(np.array(batch_delta_b), axis=0).T
                    batch_delta_b = batch_delta_b.reshape(self.b.shape)
                    self.b -= (lr * batch_delta_b)

                y_pred_batch = self.feed_forward(X_batch)
                loss = self.loss_function(y_batch, y_pred_batch) # calculate train loss for the batch
                y_pred_conv = self.conv_predicted_labels(y_pred_batch)
                accuracy = calc_accuracy(y_pred_conv, y_pred_batch) # calculate train accuracy for the batch
                train_losses.append(loss)
                train_accuracies.append(accuracy)

            # Validation
            y_pred_val = self.feed_forward(X_val)
            val_loss = self.loss_function(y_val, y_pred_val) # calculate validation loss
            y_pred_val_conv = self.conv_predicted_labels(y_pred_val) # convert labels
            val_accuracy = calc_accuracy(y_val, y_pred_val_conv) # calculate validation accuracy
            
            print("Epoch: %d / %d | Train Loss: %0.4f | Train Accuracy: %0.4f | Val Loss: %0.4f | Val Accuracy: %0.4f"%(epoch, epochs, np.mean(train_losses), np.mean(train_accuracies), val_loss, val_accuracy))
