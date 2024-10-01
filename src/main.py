import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
import os
from utils import get_data
from train_model import CV_train

def set_seed(seed):  # Define a function to set the random seed for reproducibility.
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)  # Set the seed for generating random numbers in PyTorch (CPU).
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set the hash seed for Python, affecting hash-based random operations.
    if torch.cuda.is_available():  # Check if a GPU is available.
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs if available.
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN operations.
        torch.backends.cudnn.benchmark = False  # Disable benchmarking to maintain deterministic results.

class Config:  # Define a configuration class to store various parameters.
    def __init__(self):  # Initialize the configuration parameters.
        self.datapath = './data_DC/'  # Path to the data_DC directory.
        self.save_file = './save_file/'  # Directory to save results.
        self.kfold = 5  # Number of folds for cross-validation.
        self.maskMDI = False  # Flag for masking or not (specific to the model).
        self.hidden_channels = 512  # Number of hidden channels in the model.
        self.num_heads = 4  # Number of attention heads in the model.
        self.num_layers = 4  # Number of layers in the model.
        self.self_encode_len = 256  # Length of self-encoded features.
        self.globel_random = 120  # A random seed or value (specific purpose not defined here).
        self.other_args = {'arg_name': [], 'arg_value': []}  # Dictionary for additional arguments.
        self.epochs = 2  # Number of training epochs.
        self.print_epoch = 2  # Frequency of printing logs.

def set_attr(config, param_search):  # Function to apply parameter grid to a config object.
    """
    Apply the parameter grid to the given config object and generate config instances.
    """
    param_grid = param_search  # Get the parameter search grid.
    param_keys = param_grid.keys()  # Extract parameter keys.
    param_grid_list = list(ParameterGrid(param_grid))  # Create a list of all parameter combinations.
    for param in param_grid_list:  # Iterate through each parameter combination.
        config.other_args = {'arg_name': [], 'arg_value': []}  # Reset other arguments.
        for key in param_keys:  # Set each parameter in the config object.
            setattr(config, key, param[key])  # Set the attribute in the config object.
            config.other_args['arg_name'].append(key)  # Append the parameter name.
            config.other_args['arg_value'].append(param[key])  # Append the parameter value.
        yield config  # Yield the updated config for this parameter combination.
    return 0  # Return zero when done (this line is technically unnecessary).

class Data_paths:  # Define a class to hold paths to data_DC files.
    def __init__(self):  # Initialize the paths.
        self.paths = '../data_DC/'  # Base path for data_DC files.
        self.dt = self.paths + 'd_t.csv'  # Path to a specific data_DC file (d_t).
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']  # List of additional data_DC file paths.
        self.tt = [self.paths + 't_gs.csv', self.paths + 'p_ss.csv']  # Another list of data_DC file paths.

# Define the parameter grid for the search
best_param_search = {  # Dictionary specifying the best parameters to search.
    'hidden_channels': [256],  # List of values for hidden channels.
    'num_heads': [8],  # List of values for number of heads.
    'num_layers': [6],  # List of values for number of layers.
    'CL_noise_max': [0.1],  # List of values for maximum noise (specific context needed).
}

if __name__ == '__main__':  # Entry point for the script.
    set_seed(521)  # Set the random seed for reproducibility.
    param_search = best_param_search  # Assign the parameter search grid.
    save_file = 'cross_validation_results'  # Name for the results file.

    params_all = Config()  # Instantiate the configuration object.
    param_generator = set_attr(params_all, param_search)  # Create a generator for parameter combinations.
    data_list = []  # List to hold results from each parameter combination.
    filepath = Data_paths()  # Instantiate the Data_paths object.

    while True:  # Loop to iterate over parameter combinations.
        try:
            params = next(param_generator)  # Get the next set of parameters.
        except StopIteration:  # Break the loop when all combinations have been processed.
            break

        data, y, edg_index_all = get_data(file_pair=filepath, params=params)  # Load data_DC using the provided parameters.
        data_tuple = get_data(file_pair=filepath, params=params)  # Load data_DC again for training (may need clarification).
        # Perform cross-validation and collect results
        data_idx, auc_name = CV_train(params, data_tuple)  # Train the model using cross-validation and collect AUC results.
        data_list.append(data_idx)  # Append the results to the list.

    # Concatenate cross-validation results
    if len(data_list) > 1:  # Check if more than one result exists.
        data_all = np.concatenate(tuple(x for x in data_list), axis=1)  # Concatenate results along the second dimension.
    else:
        data_all = data_list[0]  # If only one result, use it directly.

    save_path = params_all.save_file + save_file + '_MHAPR.npy'  # Define the path to save results.
    save_dir = os.path.dirname(save_path)  # Get the directory of the save path.

    if not os.path.exists(save_dir):  # Check if the directory exists.
        os.makedirs(save_dir)  # Create the directory if it does not exist.

    np.save(save_path, data_all)  # Save the concatenated results to a .npy file.
    print(auc_name)  # Print the AUC name for reference.

    # Load and analyze cross-validation results
    data_idx = np.load(params_all.save_file + save_file + '_MHAPR.npy', allow_pickle=True)  # Load the saved results.
    data_mean = data_idx[:, :, 2:].mean(0)  # Calculate the mean of the results (ignoring first two columns).
    idx_max = data_mean[:, 1].argmax()  # Find the index of the maximum value in the second column of the mean data_DC.
    print()  # Print a new line for readability.
    print('Maximum value:')  # Indicate that the following output is the maximum value found.
    print(data_mean[idx_max, :])  # Print the details of the maximum value.
