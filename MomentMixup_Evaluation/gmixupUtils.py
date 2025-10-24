from typing import List, Tuple
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import copy
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_dense_adj
import torch.nn.functional as F
import torch
import random

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        # print( data.x.shape )
        return data




def get_graphon(Res, model, coords = None):

    x_coord = (np.arange(Res) + 0.5) / Res if coords is None else coords
    y_coord = x_coord
    xx, yy = np.meshgrid(x_coord, y_coord)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    X_torch = torch.tensor(X, dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        graphon_upper = model(X_torch)
    graphon = graphon_upper.cpu().numpy().reshape(Res, Res)
    graphon = (graphon + graphon.T) / 2
    np.fill_diagonal(graphon, 0)

    return graphon



def prepare_synthetic_dataset(dataset):
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)

            data.x = F.one_hot(degs.to(torch.int64), num_classes=max_degree+1).to(torch.float)
            print(data.x.shape)


        return dataset


def prepare_dataset(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset




def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()



def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num



def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        node_x = copy.deepcopy( node_x )
        sorted_node_x = node_x[ idx, :]

        max_num = max(max_num, N)
        # if max_num < N:
        #     max_num = max(max_num, N)
        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            # added
            aligned_node_x = np.zeros((max_num, 1))
            aligned_node_x[:num_i, :] = sorted_node_x


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            #added
            aligned_node_x = aligned_node_x[:N]

    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num






def two_graphons_mixup(two_graphons, la=0.5, num_sample=20, ge='ISGL', resolution=None):

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    sample_graph_label = torch.from_numpy(label).type(torch.float32)

    Res = np.linspace(resolution[0], resolution[1], num_sample, dtype=int)
    # Res = [int(resolution[0]) for _ in range(num_sample)]
    print(Res)

    sample_graphs = []
    for i in range(num_sample):
        Res_i = Res[i]
        if ge=='ISGL' or ge=="Moment":
            # coords = np.random.uniform(0,1,Res_i)
            # coords = np.sort(coords) # None
            coords = None
            inr_1 = two_graphons[0][2]
            inr_2 = two_graphons[1][2]

            # graphon_1 = get_graphon(1000, inr_1, coords)
            # graphon_2 = get_graphon(1000, inr_2, coords)
            # new_graphon = la * graphon_1 + (1 - la) * graphon_2
            # sample_graph = simulate_graphs(w=new_graphon, num_graphs=1, num_nodes=Res_i, graph_size='fixed')[0]

            graphon_1 = get_graphon(Res_i, inr_1, coords)
            graphon_2 = get_graphon(Res_i, inr_2, coords)
            new_graphon = la * graphon_1 + (1 - la) * graphon_2
            sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)


        elif ge=='IGNR':
            gl_mlp_1 = two_graphons[0][2]
            gl_mlp_2 = two_graphons[1][2]
            graphon_1 = gl_mlp_1.get_W(Res_i, f_sample='fixed')
            graphon_2 = gl_mlp_2.get_W(Res_i, f_sample='fixed')
            new_graphon = la * graphon_1 + (1 - la) * graphon_2
            sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)

        else:
            new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]
            sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)


        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))
        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)
        num_nodes = sample_graph.shape[0]

        if num_nodes == 0:
            print('num_nodes is 0')
            continue

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
    return sample_graphs



def two_x_graphons_mixup(two_x_graphons, la=0.5, num_sample=20):

    label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
    new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
    new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    sample_graph_x = torch.from_numpy(new_x).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.x = sample_graph_x
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        # print(edge_index)
    return sample_graphs



def graphon_mixup(dataset, la=0.5, num_sample=20):
    graphons = estimate_graphon(dataset, universal_svd)

    two_graphons = random.sample(graphons, 2)
    # for label, graphon in two_graphons:
    #     print( label, graphon )
    # print(two_graphons[0][0])

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    print("new label:", label)
    # print("new graphon:", new_graphon)

    # print( label )
    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) < new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]

        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        # print(sample_graph.shape)

        # print(sample_graph)

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)
        # print(edge_index)
    return sample_graphs


def estimate_graphon(dataset, graphon_estimator):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    # print(len(all_graphs_list))

    graphons = []
    for class_label in set(y_list):
        c_graph_list = [ all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label ]

        aligned_adj_list, normalized_node_degrees, max_num, min_num = align_graphs(c_graph_list, padding=True, N=400)

        graphon_c = graphon_estimator(aligned_adj_list, threshold=0.2)

        graphons.append((np.array(class_label), graphon_c))

    return graphons



def estimate_one_graphon(aligned_adj_list: List[np.ndarray], method="universal_svd"):

    if method == "universal_svd":
        graphon = universal_svd(aligned_adj_list, threshold=0.2)
    else:
        graphon = universal_svd(aligned_adj_list, threshold=0.2)

    return graphon



def split_class_x_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    all_node_x_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)
        all_node_x_list = [graph.x.numpy()]

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list, c_node_x_list ) )

    return class_graphs


def split_class_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list ) )

    return class_graphs




def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    torch.cuda.empty_cache()
    return graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method

    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon


def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(  graph.edge_index.shape[1] )
    avg_num_nodes = sum( num_total_nodes ) / len(graphs_list)
    avg_num_edges = sum( num_total_edges ) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median( num_total_nodes ) 
    min_num_nodes = min( num_total_nodes )
    max_num_nodes = max( num_total_nodes )
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)
    std_num_nodes = np.std(num_total_nodes)

    return avg_num_nodes, max_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, std_num_nodes


from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from math import ceil
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
seed = 21
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()

        dim = num_hidden

        # Define GINConv layers with their respective MLPs
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        # Fully connected layers
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights of the model.
        """
        if isinstance(module, Linear):
            # Kaiming Normal Initialization for Linear layer weights
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # Initialize biases to zero
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        # elif isinstance(module, torch.nn.BatchNorm1d):
        #     # BatchNorm1d weights (gamma) are initialized to 1 and biases (beta) to 0 by default.
        #     # Usually, no need to change this unless you have a specific reason.
        #     torch.nn.init.constant_(module.weight, 1)
        #     torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        # Pooling layer
        # x = global_add_pool(x, batch) # Uncomment if you want to use add pooling
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training) # Dropout doesn't have weights to initialize
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

