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
        data_DC (HeteroData): Heterogeneous data_DC object.
        """
        super().__init__()

        # Dictionary to store linear layers for different node types
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            # Initialize a linear layer for each node type
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # List to store HANConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Initialize a HANConv layer for each layer
            conv = HANConv(-1, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        # Fully connected layer to produce final outputs and dropout layer
        self.fc = Linear(hidden_channels * 2, 2)
        self.dropout = torch.nn.Dropout(0.5)

        # Variables to store the best model
        self.pkl_ctl = None
        self.best_auc = 0.0

    def forward(self, data, edge_index):
        """
        Forward pass for the HAN model.

        Parameters:
        data_DC (HeteroData): Heterogeneous data_DC object containing node features.
        edge_index (Tensor): Edge indices representing the graph structure.

        Returns:
        Tensor: Output tensor after applying HAN and linear transformations.
        """
        x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()  # Copy original node features

        # Apply linear transformation and ReLU activation to node features
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        all_list = []  # List to store outputs from each convolution layer
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)  # Apply convolution
            all_list.append(x_dict.copy())  # Store the output for later use

        # Concatenate features from all layers for each node type
        for node_type in x_dict_:
            x_dict[node_type] = torch.cat([x[node_type] for x in all_list], dim=1)

        m_index, d_index = edge_index[0], edge_index[1]  # Unpack edge indices
        self.save_data = x_dict  # Save processed data_DC for later
        self.edge_index = edge_index  # Store edge indices

        # Apply dropout to the node features of type 'n1' and 'n2'
        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])

        # Compute final output by matrix multiplication
        y = Em @ Ed.t()  # Dot product between 'n1' and 'n2'
        y = y[m_index, d_index].unsqueeze(-1)  # Extract relevant entries and add a dimension
        self.y = y  # Store output for later use
        return y

    def concat_same_m_d_all(self, kf, train_idx, test_idx, y):
        """
        Save intermediate data_DC for further use.

        Parameters:
        kf (int): Current fold index for cross-validation.
        train_idx (Tensor): Indices for training samples.
        test_idx (Tensor): Indices for testing samples.
        y (Tensor): Labels for the samples.
        """
        self.train_idx = train_idx  # Store training indices
        self.test_idx = test_idx  # Store testing indices
        directory = './mid_data/'  # Specify directory for saving data_DC
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create directory if it does not exist

        # Save intermediate data_DC using joblib
        joblib.dump({
            'd_data': self.save_data['n1'],  # Store drug features
            't_data': self.save_data['n2'],  # Store target features
            'y': y.numpy(),  # Convert labels to NumPy array
            'train_idx': self.train_idx,  # Store training indices
            'test_idx': self.test_idx,  # Store testing indices
            'index': self.edge_index,  # Store edge indices
        }, f'./mid_data/{kf}-DATA.npy')  # Save data_DC to a file named by the fold index
