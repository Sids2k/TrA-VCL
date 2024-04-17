import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from dgm.models import VCL_G_Model, CommonDecoder
from dgm.modelUtils import evaluator, synthetic_pictures
from tqdm import tqdm
from ddm.testtrain import fetch_coreset
from copy import deepcopy
from ddm.vcl import AugmentedCoresetGeneration, RandomCoreset, KCenterCoreset

class VCL_G():
    def __init__(self, decoder_dimensions_common, decoder_activation_common, encoder_dimensions, encoder_activations, decoder_dims, decoder_activations, num_epochs, batch_size, coreset_formation, coreset_size = 0, augmented_coreset = False, augmented_coreset_size = 200, augmentation_iteration = 100):
        self.decoder_common = CommonDecoder(decoder_dimensions_common, decoder_activation_common)
        self.encoder_dimensions = encoder_dimensions
        self.encoder_activations = encoder_activations
        self.decoder_dims = decoder_dims
        self.decoder_activations = decoder_activations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.coreset_formation = coreset_formation
        self.coreset_size = coreset_size
        self.augmented_coreset = augmented_coreset
        self.augmented_coreset_size = augmented_coreset_size
        self.augmentation_iteration = augmentation_iteration
        if self.coreset_formation == 'random':
            self.coreset_method = RandomCoreset
        elif self.coreset_formation == 'kcenter':
            self.coreset_method = KCenterCoreset
    
    def run_vclg(self, data_gen):
        
        models = []
        results = np.zeros((data_gen.max_iter, data_gen.max_iter, 2))
        
        x_testsets = []
        x_coresets, y_coresets = [], []
        x_augmented_coresets, y_augmented_coresets = [], []
        
        for task_id in tqdm(range(data_gen.max_iter)):
            x_train, y_train, x_test, _ = data_gen.next_task()
            x_testsets.append(x_test)
            task_model = VCL_G_Model(self.encoder_dimensions, self.encoder_activations, self.decoder_dims, self.decoder_activations, self.decoder_common)
            models.append(task_model)
            if self.coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = self.coreset_method(x_coresets, y_coresets, x_train, y_train, self.coreset_size)
            task_model.train_model(self.num_epochs, torch.from_numpy(x_train), torch.from_numpy(y_train), self.batch_size)
            print("WOWO2")
            if self.augmented_coreset and self.augmented_coreset_size > 0:
                print("WOWO")
                x_augmented_coresets, y_augmented_coresets, x_train, y_train = AugmentedCoresetGeneration(x_augmented_coresets, y_augmented_coresets, x_train, y_train, self.augmented_coreset_size, self.augmentation_iteration, task_id)
                temp_x_augmented_coresets_, temp_y_augmented_coresets_ = fetch_coreset(x_augmented_coresets, y_augmented_coresets, False)
                task_model.train_model(self.num_epochs, torch.from_numpy(temp_x_augmented_coresets_), torch.from_numpy(temp_y_augmented_coresets_), self.batch_size)
            inference_models = []
            iter = 0
            for model in models:
                new_model = deepcopy(model)
                if self.coreset_size > 0:
                    temp_x_coresets, temp_y_coresets = fetch_coreset(x_coresets, y_coresets, False)
                    temp_x_coresets = temp_x_coresets[temp_y_coresets == iter]
                    temp_y_coresets = temp_y_coresets[temp_y_coresets == iter]
                    new_model.train_model(self.num_epochs, torch.from_numpy(temp_x_coresets), torch.from_numpy(temp_y_coresets), self.batch_size)
                inference_models.append(new_model)
                iter += 1
            _ = synthetic_pictures(inference_models)
            if task_id == 0:
                all_pic = _
            else:
                all_pic = np.concatenate([all_pic, _], 0)
            for test_task_id in range(len(x_testsets)):
                results[task_id, test_task_id, 0], results[task_id, test_task_id, 1] = evaluator(inference_models[test_task_id], x_testsets[test_task_id])
        
        return all_pic, results
    



