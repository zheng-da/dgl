"""
Learning Steady-States of Iterative Algorithms over Graphs
Paper: http://proceedings.mlr.press/v80/dai18a.html

"""
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(src, edge):
    # TODO should we use concat?
    return th.cat([src['in'], src['h']], 1)

def gcn_reduce(node, msgs):
    return {'accum': th.sum(msgs, 1)}

class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear1 = nn.Linear(in_feats * 2 + out_feats, out_feats)
        self.activation = activation
        # TODO what is the dimension here?
        self.linear2 = nn.Linear(out_feats, out_feats)

    def forward(self, node):
        node = th.cat([node['in'], node['accum']], 1)
        h = self.linear1(node)
        if self.activation:
            h = self.activation(h)
        return self.linear2(h)

class SSE(nn.Module):
    def __init__(self,
                 g,
                 features,
                 n_hidden,
                 activation):
        super(SSE, self).__init__()
        self.g = g
        dev = th.device(features.device.type)
        self.g.set_n_repr({'in': features,
                           'h': th.randn((g.number_of_nodes(), n_hidden), device=dev)})
        self.layer = NodeUpdateModule(features.shape[1], n_hidden, activation)

    def forward(self, vertices):
        # TODO we should support NDArray for vertex IDs.
        return self.g.pull(vertices, gcn_msg, gcn_reduce, self.layer,
                           batchable=True, writeback=False)

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu <= 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create the SSE model
    g = DGLGraph(data.graph)
    model = SSE(g,
                features,
                args.n_hidden,
                F.relu)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr * 10)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        t0 = time.time()
        randv = np.random.permutation(g.number_of_nodes())
        rand_labels = labels[randv]
        if epoch > args.warmup:
            optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
        tot_loss = 0
        for i in range(len(randv) / args.batch_size):
            data = randv[(i * args.batch_size):((i + 1) * args.batch_size)]
            label = rand_labels[(i * args.batch_size):((i + 1) * args.batch_size)]
            # TODO this isn't exactly how the model is trained.
            # We should enable the semi-supervised training.
            logits = model(data)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.detach().numpy()

            g.set_n_repr(logits, data, inplace=True)

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, tot_loss, np.mean(dur), n_edges / np.mean(dur) / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--batch-size", type=int, default=128,
            help="number of vertices in a batch")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--warmup", type=int, default=10,
            help="number of iterations to warm up with large learning rate")
    args = parser.parse_args()

    main(args)
