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

# The following function is not used in the main code
def set_seed(seed):
    """
    Sets random seeds for training.

    :param seed: Integer used for seed.
    :type seed: int
    """
    random.seed(seed)                      # Initializing a function that randomly calls a real number in [0,1) 
    np.random.seed(seed)                   # Initializing the numpy library to create random numbers uniformly distributed in [0,1)
    torch.manual_seed(seed)                # Returns a torch generator of random numbers on all cpu devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # Returns a torch generator of random numbers on all gpu devices

def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):
    """
    Pre-load adjacency matrix, map to torch device
    
    The easier tool to build a graph is networkx. It requires to insert manually nodes and edges. Then it is converted to a dgl graph in the main code.
    :param nx_graph: Graph object to pull adjacency matrix for
    :type nx_graph: networkx.OrderedGraph. This creates an indirect graph with nodes labeled with numbers.
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str (Typically used 'torch.float32')
    :return: Adjacency matrix for provided graph
    :rtype: torch.tensor
    """
    # Since networkx returns a sparse matrix, to have a regular matrix format 
    # I need to add some 0 where they're omitted with .todense()
    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device) # moving the created matrix to the specified device

    return adj_


def parse_line(file_line, node_offset):
    """
    Helper function to parse lines out of COLOR files - skips first character, which
    will be an "e" to denote an edge definition, and returns node0, node1 that define
    the edge in the line.

    :param file_line: Line to be parsed
    :type file_line: str
    :param node_offset: How much to add to account for file numbering (i.e. offset by 1)
    :type node_offset: int
    :return: Set of nodes connected by edge defined in the line (i.e. node_from, node_to)
    :rtype: int, int
    """
    # Colums are delimited by whitespace and only the 2nd and 3rd should be considered as the first one signals that the numbers label the ordered edges
    x, y = file_line.split(' ')[1:]  # skip first character - specifies each line is an edge definition
    x, y = int(x)+node_offset, int(y)+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y

def build_graph_from_color_file(fname, node_offset=-1, parent_fpath=''):
    """
    Load problem definition (graph) from COLOR file (e.g. *.col).

    :param fname: Filename of COLOR file
    :type fname: str
    :param node_offset: How much to offset node values contained in file
    :type node_offset: int
    :param parent_fpath: Path to prepend to `fname`
    :type parent_fpath: str
    :return: Graph defined in provided file
    :rtype: networkx.OrderedGraph
    """

    fpath = os.path.join(parent_fpath, fname)                                                      # Create the path to read the file

    print(f'Building graph from contents of file: {fpath}')
    with open(fpath, 'r') as f:                                                                    # Reads the entire contents of the file and stores it in the 'content' variable.
        content = f.read().strip()                                                                 #  The strip() method removes any whitespace at the beginning and end of the content.

    # Identify where problem definition starts.                                                     
    # All lines prior to this are assumed to be miscellaneous descriptions of file contents        
    # which start with "c ".                                                                       # Splits the rows using '\n' as split and creates a list containing the indices of rows starting with 'p'.
    start_idx = [idx for idx, line in enumerate(content.split('\n')) if line.startswith('p')][0]   # The value [0] of that list is then taken as the starting point.
    lines = content.split('\n')[start_idx:]  # skip comment line(s)                                # lines contains all lines from the first 'p' (inclusive) onwards.
    edges = [parse_line(line, node_offset) for line in lines[1:] if len(line) > 0]                 # edges skips the first line (the one starting with 'p') and extracts the numbers of the connected 
                                                                                                   # nodes using parse_line

    nx_temp = nx.from_edgelist(edges)                                                              # Creating graphs from lists of edges.

    ############################# MODIFIED ################################
    from collections import OrderedDict                                                            # Creating an ordered graph.
    nx_graph = nx.Graph(node_dict=OrderedDict(), edge_dict=OrderedDict())                          # This ensures that nodes and edges maintain the order in which they are added.
    #######################################################################
    
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))                                               # Import nodes into ordered graph.
    nx_graph.add_edges_from(nx_temp.edges)                                                         # Import edges into ordered graph.

    return nx_graph


# Define GNN GraphSage object
class GNNSage(nn.Module):
    """
    Basic GraphSAGE-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """

    def __init__(self, g, in_feats, hidden_size, num_classes, dropout, agg_type='mean'):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)

        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers
        :type dropout: float
        :param agg_type: Aggregation type for each SAGEConv layer. All layers will use the same agg_type
        :type agg_type: str
        """
        
        super(GNNSage, self).__init__()                                                                  # Ensures that all necessary inherited components are configured correctly.

        self.g = g
        self.num_classes = num_classes
        
        self.layers = nn.ModuleList()                                                                    # Create an (empty) list 'ModuleList', which will keep track of all the layers that the model is composed of.
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu))                 # Adds a SAGEConv layer from the dgl.nn.pytorch library with relu activation function and aggregation 'mean'
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, agg_type))                                 # Adds a SAGEConv layer from the dgl.nn.pytorch library with aggregation 'mean' but without relu activation function.
        self.dropout = nn.Dropout(p=dropout)                                                             # Dropout to avoid overfitting.

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
        h = features                                                                                      
        for i, layer in enumerate(self.layers):                                                           # Network with only one hidden layer the architecture is:
            if i != 0:                                                                                    # SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu)
                h = self.dropout(h)                                                                       # Dropout(p=dropout)
            h = layer(self.g, h)                                                                          # SAGEConv(in_feats, hidden_size, agg_type, activation=F.relu)

        return h


# Define GNN GraphConv object
class GNNConv(nn.Module):
    """
    Basic GraphConv-based GNN class object. Constructs the model architecture upon
    initialization. Defines a forward step to include relevant parameters - in this
    case, just dropout.
    """
    
    def __init__(self, g, in_feats, hidden_size, num_classes, dropout):
        """
        Initialize the model object. Establishes model architecture and relevant hypers (`dropout`, `num_classes`, `agg_type`)

        :param g: Input graph object
        :type g: dgl.DGLHeteroGraph
        :param in_feats: Size (number of nodes) of input layer
        :type in_feats: int
        :param hidden_size: Size of hidden layer
        :type hidden_size: int
        :param num_classes: Size of output layer (one node per class)
        :type num_classes: int
        :param dropout: Dropout fraction, between two convolutional layers                              
        :type dropout: float
        """
        
        super(GNNConv, self).__init__()                                                                 # Ensures that all necessary inherited components are configured correctly.
        self.g = g
        self.layers = nn.ModuleList()                                                                   # Create an (empty) list 'ModuleList', which will keep track of all the layers that the model is composed of.
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu))                         # Adds a graphconv layer from the dgl.nn.pytorch library with relu activation function.
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes))                                         # Adds a graphconv layer from the dgl.nn.pytorch library, without activation function.
        self.dropout = nn.Dropout(p=dropout)                                                            # Only one hidden layer. Dropout to avoid overfitting.

    def forward(self, features):
        """
        Define forward step of netowrk. In this example, pass inputs through convolution, apply relu
        and dropout, then pass through second convolution.

        :param features: Input node representations
        :type features: torch.tensor
        :return: Final layer representation, pre-activation (i.e. class logits)
        :rtype: torch.tensor
        """
            
        h = features
        for i, layer in enumerate(self.layers):                                                         # Network with only one hidden layer the architecture is:
            if i != 0:                                                                                  # GraphConv(in_feats, hidden_size, activation=F.relu)
                h = self.dropout(h)                                                                     # Dropout(p=dropout)
            h = layer(self.g, h)                                                                        # GraphConv(hidden_size, num_classes)
        return h


# Construct graph to learn on #
def get_gnn(g, n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Helper function to load in GNN object, optimizer, and initial embedding layer.

    :param n_nodes: Number of nodes in graph
    :type n_nodes: int
    :param gnn_hypers: Hyperparameters to provide to GNN constructor
    :type gnn_hypers: dict
    :param opt_params: Hyperparameters to provide to optimizer constructor
    :type opt_params: dict
    :param torch_device: Compute device to map computations onto (CPU vs GPU)
    :type torch_dtype: str
    :param torch_dtype: Specification of pytorch datatype to use for matrix
    :type torch_dtype: str
    :return: Initialized GNN instance, embedding layer, initialized optimizer instance
    :rtype: GNN_Conv or GNN_SAGE, torch.nn.Embedding, torch.optim.AdamW
    """

    try:                                                                                               # If the seed is specified in gnn_hypers => it is used.
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])                                                                   # Here the library function 'set_seed' is called which sets all objects to the same seed.
    except KeyError:                                                                                   # If the seed is not specified (KeyError) => seed = 0 (default).
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']                                                                        # Here all the hyperparameter values ​​are extracted from gnn_hypers.
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    print(f'Building {model} model...')
    if model == "GraphConv":                                                                           # The two previously described models are created.
        net = GNNConv(g, dim_embedding, hidden_dim, number_classes, dropout)
    elif model == "GraphSAGE":
        net = GNNSage(g, dim_embedding, hidden_dim, number_classes, dropout, agg_type)
    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)                                                       # converts the data to the specified 'torch_dtype' and moves them to device.
    embed = nn.Embedding(n_nodes, dim_embedding)                                                       # Creates the representation of the graph nodes in a space of dimension 'dim_embedding'.
    embed = embed.type(torch_dtype).to(torch_device)                                                   # converts the Embedded data to the specified 'torch_dtype' and moves them to device.

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())                                               # It is a function of the itertools library. It concatenates the parameters of net and embed into a single set of parameters.
                                                                                                       # It is useful for optimization.
    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)                             # AdamW optimizer. **opt_parms transforms the dictionary opt_parms in a variable-value set.
                                                                                                       # In this way all the parameters needed for optimization are passed, such as the lr.
    return net, embed, optimizer


# helper function for graph-coloring loss
def loss_func_mod(probs, adj_tensor):
    """
    Function to compute cost value based on soft assignments (probabilities)

    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adj_tensor: Adjacency matrix, containing internode weights
    :type adj_tensor: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: float
    """

    # Multiply probability vectors, then filter via elementwise application of adjacency matrix.
    #  Divide by 2 to adjust for symmetry about the diagonal
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2

    return loss_


# helper function for custom loss according to Q matrix
def loss_func_color_hard(coloring, nx_graph):
    """
    Function to compute cost value based on color vector (0, 2, 1, 4, 1, ...)

    :param coloring: Vector of class assignments (colors)
    :type coloring: torch.tensor
    :param nx_graph: Graph to evaluate classifications on
    :type nx_graph: networkx.OrderedGraph
    :return: Cost of provided class assignments
    :rtype: torch.tensor
    """

    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])*(u != v)
    return cost_

# THE FOLLOWING 2 FUNCTIONS WERE WRITTEN FROM SCRATCH
class SaveBestModel:
    def __init__(self, model_name='MODELNAME', best_valid_loss=float('inf')): #object initialized with best_loss = +infinite
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, metric,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss

            #print(f"\nBest validation loss: {self.best_valid_loss}")
            #print(f"\nSaving best model for epoch: {epoch+1}\n")

            # method to save a model (the state_dict: a python dictionary object that
            # maps each layer to its parameter tensor) and other useful parametrers
            # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html

            torch.save({'model' : model,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'metric': metric,
                }, 'best_model_'+self.model_name+'.pt')
            
class QuickSaveModel:
    def __init__(self, model_name='MODELNAME'):
        self.model_name = model_name
    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, metric,
    ):
        torch.save({'model' : model,
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'metric': metric,
                }, self.model_name+'.pt')

# CHANGED THE EPOCHS FROM int(1e5) TO 3000
# ADDED PROBLEM_TYPE TO SAVE THE MODELS EVERY 1000 EPOCHS
def run_gnn_training(nx_graph, graph_dgl, adj_mat, net, embed, optimizer, problem_type,
                     number_epochs=int(1e5), patience=1000, tolerance=1e-4, seed=1):
    """
    Function to run model training for given graph, GNN, optimizer, and set of hypers.
    Includes basic early stopping criteria. Prints regular updates on progress as well as
    final decision.

    :param nx_graph: Graph instance to solve
    :param graph_dgl: Graph instance to solve
    :param adj_mat: Adjacency matrix for provided graph
    :type adj_mat: torch.tensor
    :param net: GNN instance to train
    :type net: GNN_Conv or GNN_SAGE
    :param embed: Initial embedding layer
    :type embed: torch.nn.Embedding
    :param optimizer: Optimizer instance used to fit model parameters
    :type optimizer: torch.optim.AdamW
    :param number_epochs: Limit on number of training epochs to run
    :type number_epochs: int
    :param patience: Number of epochs to wait before triggering early stopping
    :type patience: int
    :param tolerance: Minimum change in cost to be considered non-converged (i.e.
        any change less than tolerance will add to early stopping count)
    :type tolerance: float

    :return: Final model probabilities, best color vector found during training, best loss found during training,
    final color vector of training, final loss of training, number of epochs used in training,
    loss_list, epoch_list
    :rtype: torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, int, list, list
    """

    # Ensure RNG seeds are reset each training run
    print(f'Function run_gnn_training(): Setting seed to {seed}')
    set_seed(seed)

    inputs = embed.weight
    # ADDED
    save_best_model = SaveBestModel(f'{problem_type}')

    # Tracking
    best_cost = torch.tensor(float('Inf'))  # high initialization
    best_loss = torch.tensor(float('Inf'))
    best_coloring = None

    # Early stopping to allow NN to train to near-completion
    prev_loss = 1.  # initial loss value (arbitrary)
    cnt = 0  # track number times early stopping is triggered

    # Initialize lists to track losses and epochs
    loss_list = []
    epoch_list = []

    # IN THE FOLLOWING THE HARD_LOSS (H_potts IN THE PAPER) IS CALLED cost_hard, 
    # WHILE THE L_potts IS THE SO CALLED loss
    # Training logic
    for epoch in range(number_epochs):

        # get soft prob assignments
        logits = net(inputs)

        # apply softmax for normalization
        probs = F.softmax(logits, dim=1)

        # get cost value with POTTS cost function
        loss = loss_func_mod(probs, adj_mat)

        # get cost based on current hard class assignments
        # update cost if applicable
        coloring = torch.argmax(probs, dim=1)
        cost_hard = loss_func_color_hard(coloring, nx_graph)

        if cost_hard < best_cost:
            best_loss = loss
            best_cost = cost_hard
            best_coloring = coloring

        # Early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
            cnt += 1
        else:
            cnt = 0
        
        # update loss tracking
        prev_loss = loss

        if cnt >= patience:
            print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
            break

        # run optimization with backpropagation
        optimizer.zero_grad()  # clear gradient for step
        loss.backward()  # calculate gradient through compute graph
        optimizer.step()  # take step, update weights
        
        # ADDED
        save_best_model(loss, epoch, net, optimizer, loss_func_mod, loss_func_color_hard)
        

        # Append current loss and epoch to lists
        loss_list.append(loss.item())
        epoch_list.append(epoch)
        
        # tracking: print intermediate loss at regular interval
        if epoch % 1000 == 0:
            print('Epoch %d | Soft Loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Discrete Cost: %.5f' % (epoch, cost_hard.item()))
            quick_save_model = QuickSaveModel(f'quick_{problem_type}_{epoch}')
            quick_save_model(loss, epoch, net, optimizer, loss_func_mod, loss_func_color_hard)

    # Print final loss
    print('Epoch %d | Final loss: %.5f' % (epoch, loss.item()))
    print('Epoch %d | Lowest discrete cost: %.5f' % (epoch, best_cost))

    # Final coloring
    final_loss = loss
    final_coloring = torch.argmax(probs, 1)
    print(f'Final coloring: {final_coloring}, soft loss: {final_loss}')

    return probs, best_coloring, best_loss, final_coloring, final_loss, epoch, loss_list, epoch_list
