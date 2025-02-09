import torch
from torch.utils.data import DataLoader
from basic_gnn import GraphDataset,collate_graphs,build_repr,get_graph_dict,get_batch_tensor,BasicGNNConv,NodeNetwork
import networkx as nx

blue,orange,green,red='#1f77b4','#ff7f0e','#2ca02c','#d62728'
mapping_dict={green:0,orange:1,blue:2,red:3}
g1=nx.Graph()
g1.add_nodes_from([(1,{'color':blue}),(2,{'color':red}),(3,{'color':green})])
g1.add_edges_from([(1,2),(2,3)])

g2=nx.Graph()
g2.add_nodes_from([(1,{'color':green}),(2,{'color':blue}),(3,{'color':green}),(0,{'color':red})])
g2.add_edges_from([(1,2),(2,3),(3,0)])

g3=nx.Graph()
g3.add_nodes_from([(1,{'color':orange}),(2,{'color':blue}),(3,{'color':green}),(0,{'color':red})])
g3.add_edges_from([(1,2),(2,3),(3,0),(1,3)])

graph_list=[get_graph_dict(graph, mapping_dict) for graph in [g1,g2,g3]]

dset=GraphDataset(graph_list)
# print(dset[0])
loader=DataLoader(dset,batch_size=2,shuffle=False,collate_fn=collate_graphs)

node_features=4
net=NodeNetwork(node_features)

results=[]

for b in loader:
    A,X,y,batch_mat=b['A'],b['X'],b['y'],b['batch_mat']
    out=net(X,A,batch_mat)
    results.append(out)