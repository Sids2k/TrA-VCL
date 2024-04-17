import pickle
import gzip
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split


class PermutedMnistGenerator():
    # Inspired from the official implementation of the paper
    def __init__(self, max_iter=10):
        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = pickle._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):

        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.arange(self.X_train.shape[1])
            np.random.shuffle(perm_inds)


            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = self.Y_train


            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = self.Y_test

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitMnistGenerator():
    # Inspired from the official implementation of the paper
    def __init__(self):
        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = pickle._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]
        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):

        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0],1 )), np.zeros((train_1_id.shape[0],1 )))).squeeze(-1)


            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0],1 )), np.zeros((test_1_id.shape[0], 1)))).squeeze(-1)

            self.cur_iter += 1
            print(next_x_train.shape, next_y_train.shape, next_x_test.shape, next_y_test.shape)
            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitNotMnistGenerator():
    def __init__(self) -> None:
        X = np.array(pickle.load(open("data/notMNIST/X_features.pickle","rb"))).reshape(-1, 28*28).astype('float32')
        Y = np.array(pickle.load(open("data/notMNIST/Y_labels.pickle","rb"))).astype('int32')
        Y = Y - 1
        X /= 255.0
        self.X_train, self.X_test, self.train_label, self.test_label = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0
        
    def get_dims(self):
        return self.X_train.shape[1], 2
    
    
    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0],1 )), np.zeros((train_1_id.shape[0],1 )))).squeeze(-1)


            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0],1 )), np.zeros((test_1_id.shape[0], 1)))).squeeze(-1)

            self.cur_iter += 1
            return next_x_train, next_y_train, next_x_test, next_y_test
        
        
class SplitMixGenerator():
    def __init__(self) -> None:
        import pickle
        X = np.array(pickle.load(open("data/notMNIST/X_features.pickle","rb"))).reshape(-1, 28*28).astype('float32')
        Y = np.array(pickle.load(open("data/notMNIST/Y_labels.pickle","rb"))).astype('int32')
        Y = Y - 1
        X /= 255.0
        self.X_train1, self.X_test1, self.train_label1, self.test_label1 = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = pickle._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]
        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0
        
    def get_dims(self):
        return self.X_train.shape[1], 2
    
    
    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label1 == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train1[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0],1 )), np.zeros((train_1_id.shape[0],1)))).squeeze(-1)

            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label1 == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test1[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0],1 )), np.zeros((test_1_id.shape[0], 1)))).squeeze(-1)

            self.cur_iter += 1
            return next_x_train, next_y_train, next_x_test, next_y_test
