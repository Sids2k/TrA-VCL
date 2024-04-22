import argparse
import numpy as np
np.random.seed(0)
from ddm.data import PermutedMnistGenerator, SplitMnistGenerator, SplitNotMnistGenerator, SplitMixGenerator
from ddm.vcl import VCL
import os 
import torch
import torch.nn.functional as F
from dgm.vcl import VCL_G
from dgm.data import DataGenerator
from dgm.modelUtils import plot_images

def main(args):
    exp_type = args.exp_type
    data_set = args.data_set
    coreset_only = args.coreset_only
    
    if exp_type == 'ddm':
        if data_set in ['mnist', 'notmnist']:
            raise ValueError('Data set not supported for DDM')
        model = VCL(args.hidden_size, args.num_epochs, args.coreset_formation, args.coreset_size, args.batch_size, args.single_head, args.augmented_coreset, args.augmented_coreset_size, args.augmentation_iteration)
        if coreset_only:
            coreset_size = args.coreset_size
            if data_set == 'permuted':
                data_gen = PermutedMnistGenerator(args.num_tasks)
            elif data_set == 'split':
                data_gen = SplitMnistGenerator()
            elif data_set == 'splitnotmnist':
                data_gen = SplitNotMnistGenerator()
            elif data_set == 'splitmix':
                data_gen = SplitMixGenerator()
            results = model.run_coreset_only(data_gen)
            print(results)
            index = 1
            while os.path.exists(f"./ddm/results/{args.coreset_formation}-{data_set}-coreset-only-{coreset_size}-v{index}.npy"):
                index += 1
            np.save(f"./ddm/results/{args.coreset_formation}-{data_set}-coreset-only-{coreset_size}-v{index}.npy", results)
        else:
            coreset_size = args.coreset_size
            if data_set == 'permuted':
                data_gen = PermutedMnistGenerator(args.num_tasks)
            elif data_set == 'split':
                data_gen = SplitMnistGenerator()
            elif data_set == 'splitnotmnist':
                data_gen = SplitNotMnistGenerator()
            elif data_set == 'splitmix':
                data_gen = SplitMixGenerator()
            results = model.run_vcl(data_gen)
            print(results)
            index = 1
            while os.path.exists(f"./ddm/results/{args.coreset_formation}-{data_set}-vcl-{coreset_size}-augmented_coreset{args.augmented_coreset}-v{index}.npy"):
                index += 1
            np.save(f"./ddm/results/{args.coreset_formation}-{data_set}-vcl-{coreset_size}-augmented_coreset{args.augmented_coreset}-v{index}.npy", results)

    elif exp_type == 'dgm':
        if data_set not in ['mnist', 'notmnist', 'mix']:
            raise ValueError('Data set not supported for DGM')
        dimX = args.dim_x
        dimH = args.dim_h
        dimZ = args.dim_z
        
        model = VCL_G([dimH, dimH, dimX], [F.relu_, torch.sigmoid], [dimX, dimH, dimH, dimH, dimZ * 2], [F.relu_, F.relu_, F.relu_, lambda x: x], [dimZ, dimH, dimH], [F.relu_, F.relu_], args.num_epochs, args.batch_size, args.coreset_formation, args.coreset_size, args.augmented_coreset, args.augmented_coreset_size, args.augmentation_iteration) 
        if data_set == 'mnist':
            data_gen = DataGenerator('mnist')
        elif data_set == 'notmnist':
            data_gen = DataGenerator('notmnist')
        elif data_set == 'mix':
            data_gen = DataGenerator('mix')
        images, losses = model.run_vclg(data_gen)

        index = 1
        while os.path.exists(f"./dgm/results/{args.coreset_formation}-{data_set}-vclg-{args.coreset_size}-augmented_coreset{args.augmented_coreset}-v{index}.npy"):
            index += 1
        np.save(f"./dgm/results/{args.coreset_formation}-{data_set}-vclg-{args.coreset_size}-augmented_coreset{args.augmented_coreset}-v{index}.npy", losses)
        with torch.no_grad():
            plot_images(images, (28, 28), 'dgm/results/figs/', f'final-{data_set}-vclg-{args.coreset_size}-augmented_coreset{args.augmented_coreset}-v{index}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-et', '--exp_type', type=str, choices=['ddm', 'dgm'], default='ddm', help='Enter the experiment type')
    parser.add_argument('-co','--coreset_only', type=bool, default=False, help='Coreset only training?')
    parser.add_argument('-ds','--data_set', type=str, choices=['split', 'permuted', 'mnist', 'notmnist', 'mix', 'splitnotmnist', 'splitmix'], default='permuted', help='Enter the dataset')
    
    
    parser.add_argument('-hs', '--hidden_size', type=list, default=[100,100], help='Hidden size for the model')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('-ne', '--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('-nt', '--num_tasks', type=int, default=10, help='Number of tasks')
    parser.add_argument('-cs', '--coreset_size', type=int, default=200, help='Coreset size')
    parser.add_argument('-sh', '--single_head', type=bool, default=True, help='Use single head or not (overridden)')
    parser.add_argument('-cf', '--coreset_formation', type=str, choices=['random', 'kcenter'], default='random', help='Coreset formation method')
    
    parser.add_argument('-ac', '--augmented_coreset', type=bool, default=False, help='Use augmented coreset or not')
    parser.add_argument('-as', '--augmented_coreset_size', type=int, default=200, help='Size of augmented coreset')
    parser.add_argument('-ai', '--augmentation_iteration', type=int, default=500, help='Number of iterations for augmentation')




    parser.add_argument('-dx', '--dim_x', type=int, default=784, help='Dimension of input')
    parser.add_argument('-dh', '--dim_h', type=int, default=500, help='Dimension of hidden h layer')
    parser.add_argument('-dz', '--dim_z', type=int, default=50, help='Dimension of z')
    
    args = parser.parse_args()
    if args.data_set in ['split', 'splitnotmnist', 'splitmix']:
        args.single_head = False
    
    main(args)
