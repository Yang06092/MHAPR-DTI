import torch
from torch_geometric.data import HeteroData  # Import HeteroData for handling heterogeneous graph data_DC.
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
    real_score, predict_score = real_score.flatten(), predict_score.flatten()  # Flatten the input arrays for processing.
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))  # Get unique sorted predicted scores.
    thresholds = sorted_predict_score[np.int32(len(sorted_predict_score) * np.arange(1, 1000) / 1000)]  # Define thresholds for ROC and PR curves.
    thresholds = np.mat(thresholds)  # Convert thresholds to a matrix.
    thresholds_num = thresholds.shape[1]  # Get the number of thresholds.

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # Create a matrix of predicted scores for each threshold.
    negative_index = np.where(predict_score_matrix < thresholds.T)  # Find indices where scores are below thresholds.
    positive_index = np.where(predict_score_matrix >= thresholds.T)  # Find indices where scores meet or exceed thresholds.
    predict_score_matrix[negative_index] = 0  # Set scores below thresholds to 0.
    predict_score_matrix[positive_index] = 1  # Set scores meeting or exceeding thresholds to 1.

    TP = predict_score_matrix.dot(real_score.T)  # Calculate True Positives (TP).
    FP = predict_score_matrix.sum(axis=1) - TP  # Calculate False Positives (FP).
    FN = real_score.sum() - TP  # Calculate False Negatives (FN).
    TN = len(real_score.T) - TP - FP - FN  # Calculate True Negatives (TN).

    fpr = FP / (FP + TN)  # Calculate False Positive Rate (FPR).
    tpr = TP / (TP + FN)  # Calculate True Positive Rate (TPR).
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T  # Prepare data_DC for ROC curve.
    ROC_dot_matrix.T[0] = [0, 0]  # Set the starting point for ROC curve.
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]  # Set the end point for ROC curve.

    x_ROC = ROC_dot_matrix[0].T  # Extract FPR values for ROC curve.
    y_ROC = ROC_dot_matrix[1].T  # Extract TPR values for ROC curve.
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])  # Calculate the Area Under the Curve (AUC).

    recall_list = tpr  # Recall is the same as TPR.
    precision_list = TP / (TP + FP)  # Calculate precision.
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T  # Prepare data_DC for Precision-Recall curve.
    PR_dot_matrix.T[0] = [0, 1]  # Set the starting point for PR curve.
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]  # Set the end point for PR curve.

    x_PR = PR_dot_matrix[0].T  # Extract recall values for PR curve.
    y_PR = PR_dot_matrix[1].T  # Extract precision values for PR curve.
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])  # Calculate the Area Under the Precision-Recall curve (AUPR).

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)  # Calculate F1 score.
    accuracy_list = (TP + TN) / len(real_score.T)  # Calculate accuracy.
    specificity_list = TN / (TN + FP)  # Calculate specificity.

    max_index = np.argmax(f1_score_list)  # Find the index of the maximum F1 score.
    f1_score = f1_score_list[max_index]  # Get the maximum F1 score.
    accuracy = accuracy_list[max_index]  # Get the corresponding accuracy.
    specificity = specificity_list[max_index]  # Get the corresponding specificity.
    recall = recall_list[max_index]  # Get the corresponding recall.
    precision = precision_list[max_index]  # Get the corresponding precision.

    # Print the calculated metrics.
    print(
        ' auc:{:.4f}, aupr:{:.4f}, f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(
            auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))

    return [real_score, predict_score, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
        ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']  # Return metrics and their names.


def get_Hdata(x1, x2, e, e_matrix, ew=None):
    """
    Create HeteroData for heterogeneous graph data_DC processing using PyTorch Geometric.

    Parameters:
    x1 (torch.Tensor): Node features for type n1.
    x2 (torch.Tensor): Node features for type n2.
    e (torch.Tensor): Edge indices.
    e_matrix (torch.Tensor): Edge matrix.
    ew (torch.Tensor, optional): Edge weights.

    Returns:
    HeteroData: Heterogeneous data_DC object for graph processing.
    """
    x1 = x1.to(device)  # Move node features for type n1 to the specified device.
    x2 = x2.to(device)  # Move node features for type n2 to the specified device.
    e = e.to(device)  # Move edge indices to the specified device.

    # Define a metadata dictionary for node types and edges.
    meta_dict = {
        'n1': {'num_nodes': x1.shape[0], 'num_features': x1.shape[1]},
        'n2': {'num_nodes': x2.shape[0], 'num_features': x2.shape[1]},
        ('n1', 'e1', 'n2'): {'edge_index': e, 'edge_weight': ew},
        ('n2', 'e1', 'n1'): {'edge_index': torch.flip(e, (0,)), 'edge_weight': ew},
    }

    data = HeteroData(meta_dict)  # Create a HeteroData object using the metadata.

    data['n1'].x = x1  # Assign node features for type n1.
    data['n2'].x = x2  # Assign node features for type n2.
    data[('n1', 'e1', 'n2')].edge_index = e  # Assign edge indices for the directed edge from n1 to n2.
    data[('n2', 'e1', 'n1')].edge_index = torch.flip(e, (0,))  # Assign reversed edge indices for the directed edge from n2 to n1.

    data['x_dict'] = {ntype: data[ntype].x for ntype in data.node_types}  # Create a dictionary of node features for all types.

    edge_index_dict = {}  # Initialize a dictionary for edge indices.
    for etype in data.edge_types:  # Loop through all edge types.
        edge_index_dict[etype] = data[etype].edge_index  # Store edge indices for each edge type.
    data['edge_dict'] = edge_index_dict  # Assign the edge index dictionary to the data_DC.

    data['edge_matrix'] = e_matrix  # Assign the edge matrix to the data_DC.
    return data.to(device)  # Move the HeteroData object to the specified device.


def get_data(file_pair, params):
    """
    Construct the dataset for model training, including adjacency matrix and node features.

    Parameters:
    file_pair (object): Contains paths to input files.
    params (object): Parameters including self-encoded length.

    Returns:
    data_DC (HeteroData): Processed heterogeneous data_DC.
    y (torch.Tensor): Labels for the edges.
    edg_index_all (torch.Tensor): Combined edge indices for positive and negative samples.
    """
    adj_matrix = pd.read_csv(file_pair.dt, header=None, index_col=None).values  # Load the adjacency matrix from a CSV file.

    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))  # Extract positive edge indices.
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)  # Convert to tensor.

    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))  # Extract negative edge indices.
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)  # Convert to tensor.

    x1 = torch.randn((adj_matrix.shape[0], params.self_encode_len))  # Generate random features for node type n1.
    x2 = torch.randn((adj_matrix.shape[1], params.self_encode_len))  # Generate random features for node type n2.

    num_pos_edges = edge_index_pos.shape[1]  # Get the number of positive edges.
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges,), dtype=torch.long)  # Randomly select negative edges.
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]  # Select the negative edges.
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)  # Combine positive and selected negative edges.

    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),  # Create labels for positive edges.
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # Create labels for negative edges.

    xe_1, xe_2 = [], []  # Initialize lists for additional features.

    if len(file_pair.dd) + len(file_pair.tt) > 0:  # Check if there are additional feature files.
        for dd_file in file_pair.dd:  # Loop through additional feature files for type n1.
            xe_1.append(pd.read_csv(dd_file, header=None, index_col=None).values)  # Load features and append to the list.
        for tt_file in file_pair.tt:  # Loop through additional feature files for type n2.
            xe_2.append(pd.read_csv(tt_file, header=None, index_col=None).values)  # Load features and append to the list.

        xe_1 = torch.tensor(np.array(xe_1).mean(0), dtype=torch.float32)  # Average the features for type n1.
        xe_2 = torch.tensor(np.array(xe_2).mean(0), dtype=torch.float32)  # Average the features for type n2.

    data = get_Hdata(xe_1, xe_2, edge_index_pos, adj_matrix)  # Create a HeteroData object using the features and edge information.

    return data, y, edg_index_all  # Return the processed data_DC, labels, and combined edge indices.
