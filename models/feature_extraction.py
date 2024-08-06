import torch
from torch_geometric.nn import Linear, HANConv
import joblib
import os


class HAN(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):
        """
        Initialize the HAN model.

        Parameters:
        hidden_channels (int): Number of hidden channels.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of layers.
        data (HeteroData): Heterogeneous data object.
        """
        super().__init__()

        # Dictionary to store linear layers for different node types
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # List to store HGTConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(-1, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        # Fully connected layer and dropout
        self.fc = Linear(hidden_channels * 2, 2)
        self.dropout = torch.nn.Dropout(0.5)

        # Variables to store the best model
        self.pkl_ctl = None
        self.best_auc = 0.0

    def forward(self, data, edge_index):
        """
        Forward pass for the HGT model.

        Parameters:
        data (HeteroData): Heterogeneous data object.
        edge_index (Tensor): Edge indices.

        Returns:
        Tensor: Output tensor after applying HGT and linear transformations.
        """
        x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()

        # Apply linear transformation and ReLU activation to node features
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())

        # Concatenate features from all layers
        for node_type in x_dict_:
            x_dict[node_type] = torch.cat([x[node_type] for x in all_list], dim=1)

        m_index, d_index = edge_index[0], edge_index[1]
        self.save_data = x_dict
        self.edge_index = edge_index

        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])

        # Compute final output
        y = Em @ Ed.t()
        y = y[m_index, d_index].unsqueeze(-1)
        self.y = y
        return y

    def concat_same_m_d_all(self, kf, train_idx, test_idx, y):
        """
        Save intermediate data for further use.

        Parameters:
        kf (int): Current fold index.
        train_idx (Tensor): Training indices.
        test_idx (Tensor): Testing indices.
        y (Tensor): Labels.
        """
        self.train_idx = train_idx
        self.test_idx = test_idx
        directory = './mid_data/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        joblib.dump({
            'd_data': self.save_data['n1'],  # drug
            't_data': self.save_data['n2'],  # target
            'y': y.numpy(),
            'train_idx': self.train_idx,
            'test_idx': self.test_idx,
            'index': self.edge_index,
        }, f'./mid_data/{kf}-DATA.npy')
