import os
import numpy as np
import pandas as pd
import torch
from .datasets import PPMI_Dataset


def read_csv_values(path, to_torch=False, sep=',', header=None,
                    index_col=None, engine='python'):

    data = pd.read_csv(path, sep=sep, header=header,
                       index_col=index_col, engine=engine)
    data = data.values

    if to_torch:
        data = torch.from_numpy(data).float()

    return data


def weighter(data):

    # Get the weights and values matrix for training

    weight_matrix = np.where(np.isnan(data)==True, 0.0,
                    np.where(np.isnan(data)==False, 1, data))
    values_matrix = np.where(np.isnan(data)==True, 0.0, data)

    return values_matrix, weight_matrix


def read_data(data, types, missing):

    # Sustitute NaN values by 0.0
    # We assume we have the real missing value mask
    true_miss_mask = np.ones([np.shape(data)[0], len(types)])
    if missing is not None:
        # The -1 is because the indexes in the csv start at 1
        true_miss_mask[missing[:, 0]-1, missing[:, 1]-1] = 0
    data_masked = np.ma.masked_where(np.isnan(data), data)

    # It is necesary to fill the data depending on the given data
    data_filler = []
    for i in range(len(types)):
        if types[i][0] == 'cat' or types[i][0] == 'ordinal':
            aux = np.unique(data[:, i]) 
            if not np.isnan(aux[0]):
                data_filler.append(aux[0])
            else:
                data_filler.append(int(0))
        else:
            data_filler.append(0.0)

    data = data_masked.filled(data_filler)

    # Construct the data matrices
    data_complete = []
    for i in range(np.shape(data)[1]):

        if types[i][0] == 'cat':
            # Get categories
            cat_data = [int(x) for x in data[:, i]]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:num_categories
            new_categories = np.arange(int(types[i][2]))
            cat_data = new_categories[indexes]
            # Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0], len(new_categories)])
            aux[np.arange(np.shape(data)[0]), cat_data] = 1
            data_complete.append(aux)
        elif types[i][0] == 'ordinal':
            # Get categories
            cat_data = [int(x) for x in data[:, i]]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:num_categories
            new_categories = np.arange(int(types[i][2]))
            cat_data = new_categories[indexes]
            # Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0], 1 + len(new_categories)])
            aux[:, 0] = 1
            aux[np.arange(np.shape(data)[0]), 1 + cat_data] = 1
            aux = np.cumsum(aux, 1)
            data_complete.append(aux)
        else:
            data_complete.append(np.transpose([data[:, i]]))

    data = np.concatenate(data_complete, 1)
    data = torch.from_numpy(data)
    # types = torch.from_numpy(types)
    true_miss_mask = torch.from_numpy(true_miss_mask)
    return data, types, true_miss_mask


def load_data_PPMI(config):

    train_dir = config.train_dir
    data = read_csv_values(os.path.join(train_dir, 'longitudinal.csv'),
                           header=0, index_col=0)

    n_patients = len(data)
    # Convert the dataset to num patients, num visits, num variables
    data = np.reshape(data, (n_patients, config.n_long_var, config.t_visits))
    data = np.swapaxes(data, 1, 2)
    X, W = weighter(data)

    # We do not convert W_train to torch because it also can be None. When it
    # does not happen we convert W_train to torch in the train loop function
    X = torch.from_numpy(X).float()
    W = torch.from_numpy(W).float()
    T = torch.from_numpy(config.t_over).float() 

    if config.static_data:
        biolog_val = read_csv_values(os.path.join(train_dir, 'Biological_VIS00.csv'), to_torch=True)
        patdemo_val = read_csv_values(os.path.join(train_dir, 'PatDemo_VIS00.csv'), to_torch=True)
        patpd_val = read_csv_values(os.path.join(train_dir, 'PatPDHist_VIS00.csv'), to_torch=True)
        stalone_val = read_csv_values(os.path.join(train_dir, 'stalone_VIS00BL_nofill.csv'), to_torch=True)

        static_vals = torch.cat((biolog_val, patdemo_val, patpd_val, stalone_val), dim=1)

        biolog_types = read_csv_values(os.path.join(train_dir, 'Biological_VIS00_types.csv'), header=0)
        patdemo_types = read_csv_values(os.path.join(train_dir, 'PatDemo_VIS00_types.csv'), header=0)
        patpd_types = read_csv_values(os.path.join(train_dir, 'PatPDHist_VIS00_types.csv'), header=0)
        stalone_types = read_csv_values(os.path.join(train_dir, 'stalone_VIS00_types.csv'), header=0)

        static_types = np.concatenate([biolog_types, patdemo_types, patpd_types, stalone_types], axis=0)

        biolog_missing = read_csv_values(os.path.join(train_dir, 'Biological_VIS00_missing.csv'))
        patdemo_missing = read_csv_values(os.path.join(train_dir, 'PatDemo_VIS00_missing.csv'))
        patpd_missing = read_csv_values(os.path.join(train_dir, 'PatPDHist_VIS00_missing.csv'))
        stalone_missing = read_csv_values(os.path.join(train_dir, 'stalone_VIS00_missing.csv'))
        static_missing = np.concatenate([biolog_missing, patdemo_missing, patpd_missing, stalone_missing], axis=0)

        # It is probably that static_types is exactly the same as the above variable
        # in that case should not be an output just to be more optimum
        static_vals_dim_ind = static_vals.shape[0]
        static_vals_dim = static_vals.shape[1]
        static_onehot, static_types, static_true_miss_mask = read_data(static_vals, static_types, static_missing)
        static_onehot_dim = static_onehot.shape[1]
    else:
        static_vals_dim_ind = None
        static_vals_dim, static_onehot_dim = None, None
        static_onehot, static_types, static_true_miss_mask = None, None, None

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_onehot_dim = static_onehot_dim
    return config, X, W, T, static_onehot, static_types, static_true_miss_mask


def load_dataset(config, only_data=False):

    if config.dataset == 'PPMI':
        data = load_data_PPMI(config)    
        config = data[0]
        data = data[1:]
        dataset = PPMI_Dataset(config, data)
        if only_data:
            return dataset
        else:
            config, dataloader = get_loader(config, dataset)
            return config, dataloader
    else:
        print('============================================')
        print('======= DATASET NOT IMPLEMENTED YET=========')
        print('============================================')

def get_loader(config, dataset):

    batch_size = int(np.round(config.batch_size * len(dataset.X)))
    config.batch_size = batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = False)
    return config, dataloader
