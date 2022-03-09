"""
saint.py: based on GraphSAINT model which is based on torch_geometric.nn.GraphConv layers
for more information see Graph Summarization with Graph Neural Networks - Technical Report and Scientific Paper
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class SAINT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=32, dropout=0.5):
        """
        if hidden_channels == 0, then a 1-layer/hop model is built,
        else a 2 layer model with hidden_channels is built
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        if hidden_channels == 0:
            self.conv1 = GraphConv(num_node_features, num_classes)
        else:
            self.conv1 = GraphConv(num_node_features, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(2 * hidden_channels, num_classes)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def forward(self, data, edge_weight=None):
        if self.hidden_channels == 0:
            x = self.conv1( data.x, data.edge_index, edge_weight )
        else:
            x1 = F.relu(self.conv1(data.x, data.edge_index, edge_weight))
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.relu(self.conv2(x1, data.edge_index, edge_weight))
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
        return F.log_softmax( x, dim=-1 )  
