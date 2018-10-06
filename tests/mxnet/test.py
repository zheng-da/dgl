import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl.backend as F

uid = mx.sym.var('src')
vid = mx.sym.var('dst')
eid = mx.sym.var('edge')
recv_vid = mx.sym.var('recv')

vertex_frame = {'h': mx.sym.var('h'),
                'in': mx.sym.var('in')}
edge_frame = {}

def msg_func1(src, edge):
    return src['in'] + 1

def msg_func2(src, edge):
    return {'m': src['in'] + 1}

def reduce_func1(node, msgs):
    return {'accum': mx.sym.sum(msgs, 1)}
def reduce_func2(node, msgs):
    return {'accum': mx.sym.sum(msgs['m'], 1)}

class NodeUpdate(gluon.HybridBlock):
    def __init__(self):
        super(NodeUpdate, self).__init__()
        self.linear = gluon.nn.Dense(10)

    def hybrid_forward(self, F, node):
        return self.linear(node)

update_func = NodeUpdate()
F.mxnet._send_and_recv(uid, vid, eid, recv_vid, vertex_frame, edge_frame,
                       msg_func2, reduce_func2, update_func)
F.mxnet._send_and_recv(uid, vid, eid, recv_vid, vertex_frame, edge_frame,
                       msg_func1, reduce_func1, update_func)
