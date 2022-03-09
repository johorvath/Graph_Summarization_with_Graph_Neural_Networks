"""
sgcn.py: SimpleGCN model based on torch_geometric.nn.SGConv layers

for more information see Graph Summarization with Graph Neural Networks - Technical Report and Scientific Paper
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGCN(torch.nn.Module):
    def __init__(self,num_features,num_classes, K = 2 ):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=K, cached = True )
       

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)