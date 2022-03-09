"""
sage.py: GraphSAGE model based on torch_geometric.nn.SAGEConv layers

for more information see Graph Summarization with Graph Neural Networks - Technical Report and Scientific Paper
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__( self, num_features,num_classes, hidden_channels = 32, dropout=0.5 ):
        """
        if hidden_channels == 0, then a 1-layer/hop model is built,
        else a 2 layer model with hidden_channels is built
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        if hidden_channels == 0:
            self.conv1 = SAGEConv(num_features, num_classes )
        else:
            self.conv1 = SAGEConv(num_features, hidden_channels )
            self.conv2 = SAGEConv(hidden_channels, num_classes )

    def forward( self, data ):
        if self.hidden_channels == 0:
            x = self.conv1( data.x, data.edge_index )
        else:
            x = self.conv1( data.x, data.edge_index )
            x = F.relu(x)
            x = F.dropout( x, training=self.training, p=self.dropout )
            x = self.conv2( x, data.edge_index )
        return F.log_softmax( x, dim=-1 )  
    
