"""
rgcn.py: rgcn model based on torch_geometric.nn.RGCNConv layers

for more information see Graph Summarization with Graph Neural Networks - Technical Report and Scientific Paper
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    def __init__(self,num_nodes,num_classes,num_relations, num_bases = None, num_blocks = None, hidden_channels = 32, dropout = 0.5 ):
        """
        if hidden_channels == 0, then a 1-layer/hop model is built,
        else a 2 layer model with hidden_channels is built
        """
        super(RGCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        if hidden_channels == 0:
            self.conv1 = RGCNConv( num_nodes, num_classes, num_relations )
        else:
            self.conv1 = RGCNConv( num_nodes, hidden_channels, num_relations )
            self.conv2 = RGCNConv( hidden_channels, num_classes, num_relations )
       

    def forward(self, data):
        if self.hidden_channels == 0:
            x = self.conv1( None, data.edge_index, data.edge_attr )
        else:
            x = self.conv1( None, data.edge_index, data.edge_attr )
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2( x, data.edge_index, data.edge_attr )
        return F.log_softmax(x, dim=1)
