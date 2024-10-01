import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from models.feature_extraction import HAN  # Import the Hierarchical Attention Network (HAN) model from a custom module.
from globel_args import device
from utils import get_metrics  # Import a utility function to calculate metrics.
from sklearn.model_selection import KFold  # Import KFold for cross-validation splitting.
import torch
import copy  # Import copy module for deep copying objects.


def train_model(data, y, edg_index_all, train_idx, test_idx, param, k_number):
    """
    Train the model with the given data_DC and parameters.

    Args:
    - data_DC (HeteroData): Graph data_DC
    - y (Tensor): Labels
    - edg_index_all (Tensor): Edge indices
    - train_idx (Tensor): Training data_DC indices
    - test_idx (Tensor): Testing data_DC indices
    - param (Config): Configuration parameters
    - k_number (int): Fold number

    Returns:
    - auc_list (list): List of AUC scores
    - auc_name (list): List of metric names
    """
    hidden_channels, num_heads, num_layers = (  # Extract model hyperparameters from the parameter object.
        param.hidden_channels, param.num_heads, param.num_layers,
    )
    epoch_param = param.epochs  # Get the number of epochs from the parameter object.

    # Build the model
    model = HAN(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data).to(device)  # Instantiate the HAN model and move it to the specified device.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)  # Set up the Adam optimizer with a learning rate and weight decay.
    data_temp = copy.deepcopy(data)  # Create a deep copy of the data_DC to avoid modifying the original during training.

    # Train the model
    auc_list = []  # Initialize a list to hold AUC scores.
    model.train()  # Set the model to training mode.
    model.param = param  # Assign the parameter configuration to the model.
    for epoch in range(1, epoch_param + 1):  # Loop through the specified number of epochs.
        optimizer.zero_grad()  # Reset the gradients for the optimizer.
        model.pkl_ctl = 'train'  # Set a control variable to indicate training mode.
        out = model(data_temp, edge_index=edg_index_all.to(device))  # Perform a forward pass through the model.
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))  # Compute the binary cross-entropy loss.
        loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update the model parameters.

        loss = loss.item()  # Get the loss value as a Python float for logging.

        if epoch % param.print_epoch == 0:  # Check if it's time to print the training status.
            model.pkl_ctl = 'test'  # Switch to test mode for evaluation.
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')  # Print the current epoch and loss.

            # Validate the model
            model.eval()  # Set the model to evaluation mode.
            with torch.no_grad():  # Disable gradient tracking during validation.
                out = model(data_temp, edge_index=edg_index_all)  # Perform a forward pass to get predictions.
                out_pred_s = out[test_idx].to('cpu').detach().numpy()  # Get predictions for the test set.
                out_pred = out_pred_s  # Store predictions.
                y_true = y[test_idx].to('cpu').detach().numpy()  # Get true labels for the test set.
                auc = roc_auc_score(y_true, out_pred)  # Calculate the AUC score.
                print('AUC:', auc)  # Print the AUC score.

                if model.best_auc < auc:  # Check if the current AUC is the best.
                    model.best_auc = auc  # Update the best AUC score.
                    model.concat_same_m_d_all(k_number, train_idx, test_idx, y)  # Store results for the best model.

                auc_idx, auc_name = get_metrics(y_true, out_pred)  # Get additional metrics for the current predictions.
                auc_idx.extend(param.other_args['arg_value'])  # Append additional argument values to the metrics.
                auc_idx.append(epoch)  # Include the current epoch in the metrics.

            auc_list.append(auc_idx)  # Add the metrics to the AUC list.
            model.train()  # Switch back to training mode.

    auc_name.extend(param.other_args['arg_name'])  # Append parameter names to the metrics list.
    return auc_list, auc_name  # Return the AUC scores and metric names.


def CV_train(param, args_tuple=()):
    """
    Perform cross-validation training.

    Args:
    - param (Config): Configuration parameters
    - args_tuple (tuple): Data tuple containing data_DC, labels, and edge indices

    Returns:
    - data_idx (ndarray): Array of AUC scores for each fold
    - auc_name (list): List of metric names
    """
    data, y, edg_index_all = args_tuple  # Unpack the data_DC tuple.
    idx = np.arange(y.shape[0])  # Create an index array for the labels.
    k_number = 1  # Initialize fold counter.
    k_fold = param.kfold  # Get the number of folds from the parameters.
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)  # Create a KFold object for cross-validation.
    kf_auc_list = []  # List to hold AUC scores for each fold.

    for train_idx, test_idx in kf.split(idx):  # Iterate through train-test splits.
        print(f'Running fold {k_number} of {k_fold}...')  # Print the current fold number.
        auc_idx, auc_name = train_model(data, y, edg_index_all, train_idx, test_idx, param, k_number)  # Train the model and get metrics.
        k_number += 1  # Increment the fold counter.
        kf_auc_list.append(auc_idx)  # Append fold AUC scores to the list.

    data_idx = np.array(kf_auc_list)  # Convert the list of AUC scores to a NumPy array.
    return data_idx, auc_name  # Return the AUC scores and metric names.
