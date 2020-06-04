import argparse
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyinstrument import Profiler

from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros

from scipy import sparse as spsp

def load_random_graph(args):
    n_nodes = args.n_nodes
    n_edges = n_nodes * 10

    row = np.random.choice(n_nodes, n_edges)
    col = np.random.choice(n_nodes, n_edges)
    spm = spsp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n_nodes, n_nodes))
    edge_index = np.vstack((spm.row, spm.col))
    edge_index = torch.LongTensor(edge_index)

    # load and preprocess dataset
    features = torch.ones((n_nodes, args.n_feats))
    labels = torch.LongTensor(np.random.choice(args.n_classes, n_nodes))
    train_mask = np.ones(shape=(n_nodes))
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)
    else:
        train_mask = torch.ByteTensor(train_mask)
    print("""----Data statistics------'
      #Edges %d
      #Train samples %d""" % (n_edges, train_mask.int().sum().item()))

    return edge_index, features, labels, train_mask

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, adj, feat):
        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, self.weight)
            rst = adj.matmul(feat)
        else:
            rst = adj.matmul(feat) @ self.weight

        rst = rst + self.bias

        return rst

class GCN(nn.Module):
    def __init__(self,
                 adj,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.adj = adj
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.adj, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def main(args):
    if args.gpu >= 0:
        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    device = torch.device(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    edge_index, features, labels, train_mask = load_random_graph(args)
    adj = SparseTensor(row=edge_index[0].to(device), col=edge_index[1].to(device))
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)

    # create GCN model
    model = GCN(adj=adj,
                in_feats=args.n_feats,
                n_hidden=args.n_hidden,
                n_classes=args.n_classes,
                n_layers=args.n_layers,
                activation=F.relu,
                dropout=args.dropout).to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    profiler = Profiler()
    profiler.start()
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)
            print('Training time: {:.4f}'.format(np.mean(dur)))
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--n-nodes", type=int, default=10000000,
                        help="Number of nodes in the random graph")
    parser.add_argument("--n-feats", type=int, default=100,
                        help="Number of input node features")
    parser.add_argument("--n-classes", type=int, default=10,
                        help="Number of node classes")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
