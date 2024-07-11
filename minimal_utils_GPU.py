import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os

from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
from itertools import chain

# Known chromatic numbers for specified problems (from references)
chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen13_13.col': 13,
    # Citations graphs
    'cora.cites': 5,
    'citeseer.cites': 6,
    'pubmed.cites': 8
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):
    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device)
    return adj_

def parse_line(file_line, node_offset):
    x, y = file_line.split(' ')[1:]  # skip first character - specifies each line is an edge definition
    x, y = int(x)+node_offset, int(y)+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y

def build_graph_from_color_file(fname, node_offset=-1, parent_fpath=''):
    fpath = os.path.join(parent_fpath, fname)
    print(f'Building graph from contents of file: {fpath}')
    with open(fpath, 'r') as f:
        content = f.read().strip()

    start_idx = [idx for idx, line in enumerate(content.split('\n')) if line.startswith('p')][0]
    lines = content.split('\n')[start_idx:]  # skip comment line(s)
    edges = [parse_line(line, node_offset) for line in lines[1:] if len(line) > 0]

    nx_temp = nx.from_edgelist(edges)

    from collections import OrderedDict
    nx_graph = nx.Graph(node_dict=OrderedDict(), edge_dict=OrderedDict())
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)

    return nx_graph

# Define GNN GraphSage object
class GNNSage(nn.Module):
    def __init__(self, g, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):
        super(GNNSage, self).__init__()
        self.g = g
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

# Define GNN GraphConv object
class GNNConv(nn.Module):
    def __init__(self, g, in_feats, hidden_size, num_classes, dropout):
        super(GNNConv, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu))
        self.layers.append(GraphConv(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

# Construct graph to learn on #
def get_gnn(g, n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    print(f'Building {model} model...')
    if model == "GraphConv":
        net = GNNConv(g, dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(g, dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    params = chain(net.parameters(), embed.parameters())
    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

    return net, embed, optimizer

# helper function for graph-coloring loss
def loss_func_mod(probs, adj_tensor):
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2
    return loss_

# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, nx_graph):
    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])*(u != v)
    return cost_

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_gnn_training(nx_graph, graph_dgl, adj_mat, net, embed, optimizer,
                     number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    inputs = embed.weight

    best_cost = torch.tensor(float('Inf'))  # high initialization
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None

    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    for epoch in range(number_epochs):
        logits = net(inputs)
        probs = F.softmax(logits, dim=1)
        loss = loss_func_mod(probs, adj_mat)
        coloring = torch.argmax(probs, dim=1)
        cost_hard = loss_func_color_hard(coloring, nx_graph)

        if cost_hard < best_cost:
            best_loss = loss
            best_cost = cost_hard
            best_coloring = coloring

        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('Epoch %d | Soft Loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Discrete Cost: %.5f' % (epoch, cost_hard.item()))

    print('Epoch %d | Final loss: %.5f' % (epoch, loss.item()))
    print('Epoch %d | Lowest discrete cost: %.5f' % (epoch, best_cost))

    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final coloring: {final_coloring}, soft loss: {final_loss}')

    return probs, best_coloring, best_loss, final_coloring, final_loss, epoch

# Main code to
