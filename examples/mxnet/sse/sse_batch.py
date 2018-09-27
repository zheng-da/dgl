"""
Learning Steady-States of Iterative Algorithms over Graphs
Paper: http://proceedings.mlr.press/v80/dai18a.html

"""
import argparse
import numpy as np
import time
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(src, edge):
    # TODO should we use concat?
    return mx.nd.concat(src['in'], src['h'], dim=1)

def gcn_reduce(node, msgs):
    return {'accum': mx.nd.sum(msgs, 1)}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, alpha=0.9):
        super(NodeUpdate, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation=activation)
        # TODO what is the dimension here?
        self.linear2 = gluon.nn.Dense(out_feats)
        self.alpha = alpha

    def forward(self, node):
        tmp = mx.nd.concat(node['in'], node['accum'], dim=1)
        hidden = self.linear2(self.linear1(tmp))
        return node['h'] * (1 - self.alpha) + self.alpha * hidden

class SSEUpdateHidden(gluon.Block):
    def __init__(self,
                 g,
                 features,
                 n_hidden,
                 activation):
        super(SSEUpdateHidden, self).__init__()
        self.g = g
        self.g.set_n_repr({'in': features,
                           'h': mx.nd.random.normal(shape=(g.number_of_nodes(), n_hidden), ctx=features.context)})
        self.layer = NodeUpdate(n_hidden, activation)

    def forward(self, vertices):
        if vertices is None:
            self.g.update_all(gcn_msg, gcn_reduce, self.layer,
                    batchable=True)
            return self.g.get_n_repr()
        else:
            # TODO we should support NDArray for vertex IDs.
            vs = vertices.asnumpy()
            return self.g.pull(vs, gcn_msg, gcn_reduce, self.layer,
                    batchable=True, writeback=False)

class SSEPredict(gluon.Block):
    def __init__(self, update_hidden, out_feats):
        super(SSEPredict, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation='relu')
        self.linear2 = gluon.nn.Dense(out_feats)
        self.update_hidden = update_hidden

    def forward(self, vertices):
        hidden = self.update_hidden(vertices)
        return self.linear2(self.linear1(hidden))

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu <= 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        features = features.as_in_context(mx.gpu(0))
        labels = labels.as_in_context(mx.gpu(0))
        mask = mask.as_in_context(mx.gpu(0))
        ctx = mx.gpu(0)

    # create the SSE model
    g = DGLGraph(data.graph)
    update_hidden = SSEUpdateHidden(g, features, args.n_hidden, 'relu')
    model = SSEPredict(update_hidden, args.n_hidden)
    model.initialize(ctx=ctx)

    # use optimizer
    num_batches = int(g.number_of_nodes() / args.batch_size)
    scheduler = mx.lr_scheduler.CosineScheduler(args.n_epochs * num_batches,
            args.lr * 10, 0, 0, args.lr/5)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr,
        'lr_scheduler': scheduler})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        # compute vertex embedding.
        update_hidden(None)

        t0 = time.time()
        randv = np.random.permutation(g.number_of_nodes())
        rand_labels = labels[randv]
        data_iter = mx.io.NDArrayIter(data=mx.nd.array(randv, dtype='int32'), label=rand_labels,
                                      batch_size=args.batch_size)
        tot_loss = 0
        for batch in data_iter:
            # TODO this isn't exactly how the model is trained.
            # We should enable the semi-supervised training.
            with mx.autograd.record():
                logits = model(batch.data[0])
                loss = mx.nd.softmax_cross_entropy(logits, batch.label[0])
            loss.backward()
            trainer.step(batch.data[0].shape[0])
            tot_loss += loss.asnumpy()[0]

            g.set_n_repr(logits, batch.data[0], inplace=True)

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
