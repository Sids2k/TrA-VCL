import numpy as np
from ddm.testtrain import get_scores, concatenate_results, fetch_coreset
from ddm.testtrain import NN
from pygan._torch.gan_image_generator import GANImageGenerator
import torch
device_for_gan = "cuda:0" if torch.cuda.is_available() else "cpu"
import cv2
import os
import shutil

def RandomCoreset(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Inspired from the official implementation of the paper
    idx = np.random.choice(x_train.shape[0], coreset_size, False)
    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train 

def KCenterCoreset(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Inspired from the official implementation of the paper
    dists = np.full(x_train.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train, current_id)
    idx = [ current_id ]

    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)

    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train


def AugmentedCoresetGeneration(x_augmented_coresets, y_augmented_coresets, x_train, y_train, augmented_coreset_size = 100, augmentation_iteration = 100, task_id = None):
    temp_augmented_x = []
    temp_augmented_y = []
    if task_id is not None:
        unique_labels = np.unique(y_train[y_train == task_id]).astype(int)
    else:
        unique_labels = np.unique(y_train).astype(int)
    x_train_parts = []
    y_train_parts = []
    
    for label in unique_labels:
        x_train_parts.append(x_train[y_train == label])
        y_train_parts.append(y_train[y_train == label])
    
    if task_id is None:
        for i in range(len(x_train_parts)):
            if task_id is not None:
                i = task_id
            folder_path = f"data/temporary_folder/{i}"
            os.makedirs(folder_path, exist_ok=True)
            for j in range(len(x_train_parts[i])):
                image_path = f"{folder_path}/{j}.png"
                normalized_image = (x_train_parts[i][j].reshape(28, 28) * 255).astype(np.uint8)
                cv2.imwrite(image_path, normalized_image)
    else:
        folder_path = f"data/temporary_folder/{task_id}"
        os.makedirs(folder_path, exist_ok=True)
        for j in range(len(x_train_parts[0])):
            image_path = f"{folder_path}/{j}.png"
            normalized_image = (x_train_parts[0][j].reshape(28, 28) * 255).astype(np.uint8)
            cv2.imwrite(image_path, normalized_image)
    
    for i in range(len(x_train_parts)):
        if task_id is not None:
            i = task_id
        gan_image_generator = GANImageGenerator(
            dir_list=[f"data/temporary_folder/{i}",], width=28, height=28, channel=1, batch_size=augmented_coreset_size, learning_rate=1e-06, ctx=device_for_gan,)
        if task_id is not None:
            print("Augmenting dataset for class ", unique_labels[0])
        else:
            print("Augmenting dataset for class ", unique_labels[i])
        gan_image_generator.learn(iter_n=augmentation_iteration, k_step=10,)
        arr = gan_image_generator.GAN.generative_model.draw()
        temparr = []
        for _ in range(arr.shape[0]):
            temparr.append(cv2.resize(arr[_][0].detach().cpu().numpy(), (28, 28)).flatten()/255)
        arr = np.array(temparr)
        
        if len(temp_augmented_x) == 0:
            temp_augmented_x = arr
            if task_id is not None:
                temp_augmented_y = [unique_labels[0]] * arr.shape[0]
            else:
                temp_augmented_y = [unique_labels[i]] * arr.shape[0]
        else:
            temp_augmented_x = np.concatenate((temp_augmented_x,arr), axis = 0)
            if task_id is not None:
                temp_augmented_y = np.concatenate((temp_augmented_y, [unique_labels[0]] * arr.shape[0]), axis = 0)
            else:
                temp_augmented_y = np.concatenate((temp_augmented_y, [unique_labels[i]] * arr.shape[0]), axis = 0)
        if task_id is not None:
            break
    x_augmented_coresets.append(np.array(temp_augmented_x))
    y_augmented_coresets.append(np.array(temp_augmented_y))
    
    shutil.rmtree("data/temporary_folder")

    return x_augmented_coresets, y_augmented_coresets, x_train, y_train

def update_distance(dists, x_train, current_id):
    # Inspired from the official implementation of the paper
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists


class VCL():
    def __init__(self, hidden_size, num_epochs, coreset_formation, coreset_size=0, batch_size=None, single_head=True, augmented_coreset = False, augmented_coreset_size = 200, augmentation_iteration = 100):
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.coreset_formation = coreset_formation
        self.coreset_size = coreset_size
        self.batch_size = batch_size
        self.single_head = single_head
        if self.coreset_formation == 'random':
            self.coreset_method = RandomCoreset
        elif self.coreset_formation == 'kcenter':
            self.coreset_method = KCenterCoreset
        self.augmented_coreset = augmented_coreset
        self.augmented_coreset_size = augmented_coreset_size
        self.augmentation_iteration = augmentation_iteration
    
    def run_coreset_only(self, data_gen):
        in_dim, out_dim = data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []
        all_acc = np.array([])

        for task_id in range(data_gen.max_iter):
            x_train, y_train, x_test, y_test = data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            head = 0 if self.single_head else task_id
            bsize = x_train.shape[0] if (self.batch_size is None) else self.batch_size

            if task_id == 0:
                mf_model = NN(in_dim, self.hidden_size, out_dim, x_train.shape[0], single_head = self.single_head, prev_means=None, type = 'BNN')

            if self.coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = self.coreset_method(x_coresets, y_coresets, x_train, y_train, self.coreset_size)


            mf_model.save_weights()

            acc = get_scores(mf_model, x_testsets, y_testsets, self.num_epochs, self.single_head, x_coresets, y_coresets, self.batch_size)

            all_acc = concatenate_results(acc, all_acc)

            mf_model.load_weights()
            mf_model.clean_weights()

            if not self.single_head:
                mf_model.create_head()

        return all_acc
    
    def run_vcl(self, data_gen):
        in_dim, out_dim = data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []
        x_augmented_coresets, y_augmented_coresets = [], []
        all_acc = np.array([])

        for task_id in range(data_gen.max_iter):
            x_train, y_train, x_test, y_test = data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            head = 0 if self.single_head else task_id
            bsize = x_train.shape[0] if (self.batch_size is None) else self.batch_size

            if task_id == 0:
                ml_model = NN(in_dim, self.hidden_size, out_dim, x_train.shape[0], type='NN')
                ml_model.train(x_train, y_train, task_id, self.num_epochs, bsize)
                mf_weights = ml_model.get_weights()
                mf_model = NN(in_dim, self.hidden_size, out_dim, x_train.shape[0], single_head = self.single_head, prev_means=mf_weights, type='BNN')
                
            if self.coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = self.coreset_method(x_coresets, y_coresets, x_train, y_train, self.coreset_size)
            
            mf_model.train(x_train, y_train, head, self.num_epochs, bsize)

            if self.augmented_coreset and self.augmented_coreset_size > 0:
                mf_model.update_prior()
                x_augmented_coresets, y_augmented_coresets, x_train, y_train = AugmentedCoresetGeneration(x_augmented_coresets, y_augmented_coresets, x_train, y_train, self.augmented_coreset_size, self.augmentation_iteration)
                temp_x_augmented_coresets, temp_y_augmented_coresets = fetch_coreset(x_augmented_coresets, y_augmented_coresets, self.single_head)
                mf_model.train(temp_x_augmented_coresets, temp_y_augmented_coresets, head, self.num_epochs, bsize)
            
            mf_model.update_prior()

            mf_model.save_weights()

            acc = get_scores(mf_model, x_testsets, y_testsets, self.num_epochs, self.single_head, x_coresets, y_coresets, self.batch_size)
            all_acc = concatenate_results(acc, all_acc)

            mf_model.load_weights()
            mf_model.clean_weights()


            if not self.single_head:
                mf_model.create_head()

        return all_acc


