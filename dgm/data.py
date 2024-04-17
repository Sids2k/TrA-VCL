import pickle
import gzip
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch

class DataGenerator():
    def __init__(self, type = 'mnist', max_iter=10):
        if type == 'mnist':
            with gzip.open('data/mnist.pkl.gz', 'rb') as file:
                u = pickle._Unpickler(file)
                u.encoding = 'latin1'
                p = u.load()
                train_set, valid_set, test_set = p

            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.Y_train = np.hstack((train_set[1], valid_set[1]))
            self.X_test = test_set[0]
            self.Y_test = test_set[1]
        elif type == 'notmnist':
            X = np.array(pickle.load(open("data/notMNIST/X_features.pickle","rb"))).reshape(-1, 28*28).astype('float32')
            Y = np.array(pickle.load(open("data/notMNIST/Y_labels.pickle","rb"))).astype('int32')
            Y = Y - 1
            X /= 255.0
            self.X_train, self.X_test, self.train_label, self.test_label = train_test_split(X, Y, test_size=0.2, random_state=42)
        elif type == 'mix':
            with gzip.open('data/mnist.pkl.gz', 'rb') as file:
                u = pickle._Unpickler(file)
                u.encoding = 'latin1'
                p = u.load()
                train_set, valid_set, test_set = p
            X = np.array(pickle.load(open("data/notMNIST/X_features.pickle","rb"))).reshape(-1, 28*28).astype('float32')
            Y = np.array(pickle.load(open("data/notMNIST/Y_labels.pickle","rb"))).astype('int32')
            Y = Y - 1
            X /= 255.0
            
            X = X[Y < 5]
            Y = Y[Y < 5]
            Y = Y + 5
            X_train = np.vstack((train_set[0], valid_set[0], test_set[0]))
            Y_train = np.hstack((train_set[1], valid_set[1], test_set[1]))
            X_train = X_train[Y_train < 5]
            Y_train = Y_train[Y_train < 5]
            
            X_train = np.vstack((X_train, X))
            Y_train = np.hstack((Y_train, Y))
            
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
            
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            next_x_train = deepcopy(self.X_train)
            next_y_train = self.Y_train


            next_x_test = deepcopy(self.X_test)
            next_y_test = self.Y_test

            self.cur_iter += 1
            return next_x_train, next_y_train, next_x_test, next_y_test

