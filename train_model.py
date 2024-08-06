import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from models.feature_extraction import HAN
from globel_args import device
from utils import get_metrics
from sklearn.model_selection import KFold
import torch
import copy



def train_model(data, y, edg_index_all, train_idx, test_idx, param, k_number):
    """
    Train the model with the given data and parameters.

    Args:
    - data (HeteroData): Graph data
    - y (Tensor): Labels
    - edg_index_all (Tensor): Edge indices
    - train_idx (Tensor): Training data indices
    - test_idx (Tensor): Testing data indices
    - param (Config): Configuration parameters
    - k_number (int): Fold number

    Returns:
    - auc_list (list): List of AUC scores
    - auc_name (list): List of metric names
    """
    hidden_channels, num_heads, num_layers = (
        param.hidden_channels, param.num_heads, param.num_layers,
    )
    epoch_param = param.epochs

    # Build the model
    model = HAN(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)
    data_temp = copy.deepcopy(data)

    # Train the model
    auc_list = []
    model.train()
    model.param = param
    for epoch in range(1, epoch_param + 1):
        optimizer.zero_grad()
        model.pkl_ctl = 'train'
        out = model(data_temp, edge_index=edg_index_all.to(device))
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))
        loss.backward()
        optimizer.step()
        loss = loss.item()

        if epoch % param.print_epoch == 0:
            model.pkl_ctl = 'test'
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            # Validate the model
            model.eval()
            with torch.no_grad():
                out = model(data_temp, edge_index=edg_index_all)
                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(y_true, out_pred)
                print('AUC:', auc)

                if model.best_auc < auc:
                    model.best_auc = auc
                    model.concat_same_m_d_all(k_number, train_idx, test_idx, y)

                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)

            auc_list.append(auc_idx)
            model.train()

    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name


def CV_train(param, args_tuple=()):
    """
    Perform cross-validation training.

    Args:
    - param (Config): Configuration parameters
    - args_tuple (tuple): Data tuple containing data, labels, and edge indices

    Returns:
    - data_idx (ndarray): Array of AUC scores for each fold
    - auc_name (list): List of metric names
    """
    data, y, edg_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)
    kf_auc_list = []

    for train_idx, test_idx in kf.split(idx):
        print(f'Running fold {k_number} of {k_fold}...')
        auc_idx, auc_name = train_model(data, y, edg_index_all, train_idx, test_idx, param, k_number)
        k_number += 1
        kf_auc_list.append(auc_idx)

    data_idx = np.array(kf_auc_list)
    return data_idx, auc_name