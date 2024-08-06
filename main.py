import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
import os
from utils import get_data
from train_model import CV_train


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    def __init__(self):
        self.datapath = './data/'
        self.save_file = './save_file/'
        self.kfold = 5
        self.maskMDI = False
        self.hidden_channels = 512
        self.num_heads = 4
        self.num_layers = 4
        self.self_encode_len = 256
        self.globel_random = 120
        self.other_args = {'arg_name': [], 'arg_value': []}
        self.epochs = 2
        self.print_epoch = 2


def set_attr(config, param_search):
    """
    Apply the parameter grid to the given config object and generate config instances.
    """
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for key in param_keys:
            setattr(config, key, param[key])
            config.other_args['arg_name'].append(key)
            config.other_args['arg_value'].append(param[key])
        yield config
    return 0


class Data_paths:
    def __init__(self):
        self.paths = './data_DC/'
        self.dt = self.paths + 'd_t.csv'
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']
        self.tt = [self.paths + 't_gs.csv', self.paths + 'p_ss.csv']


# Define the parameter grid for the search
best_param_search = {
    'hidden_channels': [256],
    'num_heads': [8],
    'num_layers': [6],
    'CL_noise_max': [0.1],
}

if __name__ == '__main__':
    set_seed(521)
    param_search = best_param_search
    save_file = 'cross_validation_results'

    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []
    filepath = Data_paths()

    while True:
        try:
            params = next(param_generator)
        except StopIteration:
            break

        data, y, edg_index_all = get_data(file_pair=filepath, params=params)
        data_tuple = get_data(file_pair=filepath, params=params)
        # Perform cross-validation and collect results
        data_idx, auc_name = CV_train(params, data_tuple)
        data_list.append(data_idx)

    # Concatenate cross-validation results
    if len(data_list) > 1:
        data_all = np.concatenate(tuple(x for x in data_list), axis=1)
    else:
        data_all = data_list[0]

    np.save(params_all.save_file + save_file + '_MHAPR.npy', data_all)
    data_all
    print(auc_name)

    # Load and analyze cross-validation results
    data_idx = np.load(params_all.save_file + save_file + '_MHAPR.npy', allow_pickle=True)
    data_mean = data_idx[:, :, 2:].mean(0)
    idx_max = data_mean[:, 1].argmax()
    print()
    print('Maximum value:')
    print(data_mean[idx_max, :])
