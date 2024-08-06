import torch
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from globel_args import device


def get_metrics(real_score, predict_score):
    """
    Calculate performance metrics for binary classification models, including ROC curve, AUC, PR curve, and more.

    Parameters:
    real_score (np.array): Actual scores.
    predict_score (np.array): Predicted scores.

    Returns:
    list: Calculated metrics.
    list: Names of the metrics.
    """
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    print(
        ' auc:{:.4f}, aupr:{:.4f}, f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(
            auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))

    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def get_Hdata(x1, x2, e, e_matrix, ew=None):
    """
    Create HeteroData for heterogeneous graph data processing using PyTorch Geometric.

    Parameters:
    x1 (torch.Tensor): Node features for type n1.
    x2 (torch.Tensor): Node features for type n2.
    e (torch.Tensor): Edge indices.
    e_matrix (torch.Tensor): Edge matrix.
    ew (torch.Tensor, optional): Edge weights.

    Returns:
    HeteroData: Heterogeneous data object for graph processing.
    """
    x1 = x1.to(device)
    x2 = x2.to(device)
    e = e.to(device)

    meta_dict = {
        'n1': {'num_nodes': x1.shape[0], 'num_features': x1.shape[1]},
        'n2': {'num_nodes': x2.shape[0], 'num_features': x2.shape[1]},
        ('n1', 'e1', 'n2'): {'edge_index': e, 'edge_weight': ew},
        ('n2', 'e1', 'n1'): {'edge_index': torch.flip(e, (0,)), 'edge_weight': ew},
    }

    data = HeteroData(meta_dict)

    data['n1'].x = x1
    data['n2'].x = x2
    data[('n1', 'e1', 'n2')].edge_index = e
    data[('n2', 'e1', 'n1')].edge_index = torch.flip(e, (0,))

    data['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}

    edge_index_dict = {}
    for etype in data.edge_types:
        edge_index_dict[etype] = data[etype].edge_index
    data['edge_dict'] = edge_index_dict

    data['edge_matrix'] = e_matrix
    return data.to(device)


def get_data(file_pair, params):
    """
    Construct the dataset for model training, including adjacency matrix and node features.

    Parameters:
    file_pair (object): Contains paths to input files.
    params (object): Parameters including self-encoded length.

    Returns:
    data (HeteroData): Processed heterogeneous data.
    y (torch.Tensor): Labels for the edges.
    edg_index_all (torch.Tensor): Combined edge indices for positive and negative samples.
    """
    adj_matrix = pd.read_csv(file_pair.dt, header=None, index_col=None).values

    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    x1 = torch.randn((adj_matrix.shape[0], params.self_encode_len))
    x2 = torch.randn((adj_matrix.shape[1], params.self_encode_len))

    num_pos_edges = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges,), dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)

    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)

    xe_1, xe_2 = [], []

    if len(file_pair.dd) + len(file_pair.tt) > 0:
        for dd_file in file_pair.dd:
            xe_1.append(pd.read_csv(dd_file, header=None, index_col=None).values)
        for tt_file in file_pair.tt:
            xe_2.append(pd.read_csv(tt_file, header=None, index_col=None).values)

        xe_1 = torch.tensor(np.array(xe_1).mean(0), dtype=torch.float32)
        xe_2 = torch.tensor(np.array(xe_2).mean(0), dtype=torch.float32)

    data = get_Hdata(xe_1, xe_2, edge_index_pos, adj_matrix)

    return data, y, edg_index_all
