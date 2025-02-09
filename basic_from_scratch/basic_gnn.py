'''
An implementation of a basic GNN model, C:Sebastian Raschka
'''

import networkx as nx
import torch
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch.nn.functional as F

class NodeNetwork(torch.nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.conv1=BasicGNNConv(input_features,32)
        self.conv2=BasicGNNConv(32,32)
        self.fc1=torch.nn.Linear(32,16)
        self.out=torch.nn.Linear(16,2)
        self.pool=global_sum_pool()
    
    def forward(self,X,A,batch_mat):
        X=F.relu(self.conv1(X,A))
        X=F.relu(self.conv2(X,A))
        output=self.pool(X,batch_mat)
        print(output.size(),batch_mat.size())
        output=F.relu(self.fc1(output)) 
        output=self.out(output)
        return F.softmax(output,dim=1)
    
class BasicGNNConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.w1=Parameter(torch.rand(in_channels,out_channels,dtype=torch.float32))
        self.w2=Parameter(torch.rand(in_channels,out_channels,dtype=torch.float32))
        self.bias=Parameter(torch.rand(out_channels,dtype=torch.float32))
        self.reset_parameters()
    def reset_parameters(self):
        pass
    def forward(self,X,A):
        root=torch.matmul(X,self.w1)
        msg=torch.matmul(A,X)
        msg=torch.matmul(msg,self.w2)
        output=root+msg+self.bias
        return output
    
class global_sum_pool(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X,batch_mat):
        if batch_mat is None or batch_mat.dim()==1:
            return torch.sum(X,dim=0).unsqueeze(0)
        else:
            return torch.matmul(batch_mat,X)

def get_batch_tensor(graph_sizes):
    starts=[sum(graph_sizes[:i]) for i in range(len(graph_sizes))]
    stops=[starts[i] + graph_sizes[i] for i in range(len(graph_sizes))]

    total_len=sum(graph_sizes)
    batch_size=len(graph_sizes)
    batch_mat=torch.zeros((batch_size,total_len)).float()
    for i,starts_and_stops in enumerate(zip(starts,stops)):
        start=starts_and_stops[0]
        stop=starts_and_stops[1]
        batch_mat[i,start:stop]=1
    return batch_mat

def collate_graphs(batch):
    adj_mat=[graph['A'] for graph in batch]
    sizes=[A.size(0) for A in adj_mat]
    total_size=sum(sizes)

    batch_mat=get_batch_tensor(sizes)

    feature_mat=torch.cat([graph['X'] for graph in batch],dim=0)

    labels=torch.cat([graph['y'] for graph in batch])

    batch_adj=torch.zeros((total_size,total_size),dtype=torch.float32)
    accum=0
    for adj in adj_mat:
        g_size=adj.shape[0]
        batch_adj[accum:accum+g_size,accum:accum+g_size]=adj
        accum+=g_size
    repr_and_label={'A':batch_adj,'X':feature_mat,'y':labels,'batch_mat':batch_mat}
    return repr_and_label

def build_repr(G,mapping_dict):
    one_hot_idx=np.array([mapping_dict[v] for v in nx.get_node_attributes(G,'color').values()])
    # print(one_hot_idx)
    one_hot_enc=np.zeros((len(one_hot_idx),len(mapping_dict)))
    one_hot_enc[np.arange(one_hot_idx.size),one_hot_idx]=1

    return one_hot_enc

def get_graph_dict(G,mapping_dict):
    A=torch.from_numpy(np.asarray(nx.adjacency_matrix(G).todense())).float()
    X=torch.from_numpy(build_repr(G,mapping_dict)).float()
    y=torch.tensor([[1,0]]).float()

    return {'A':A,'X':X,'y':y,'batch_mat':None}

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self,graphs):
        self.graphs=graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self,idx):
        return self.graphs[idx]
    
