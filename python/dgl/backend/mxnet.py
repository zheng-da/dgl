from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import scipy.sparse
import ctypes

from .._ffi.base import _LIB, check_call, c_array
from .._ffi.runtime_ctypes import TVMType, TVMContext, TVMArray
from .._ffi.runtime_ctypes import TypeCode, tvm_shape_index_t

# Tensor types
Tensor = mx.nd.NDArray
SparseTensor = mx.nd.sparse.CSRNDArray

# Data types
float16 = np.float16
float32 = np.float32
float64 = np.float64
uint8 = np.uint8
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

# Operators
tensor = mx.nd.array
#sparse_tensor = th.sparse.FloatTensor
sum = F.sum
max = F.max

def astype(a, ty):
    return F.cast(a, ty)

def asnumpy(a):
    return a.asnumpy()

def from_numpy(np_data):
    return mx.nd.array(np_data)

def pack(tensors):
    return F.concat(*tensors, dim=0)

def unpack(x, indices_or_sections=1):
    return th.split(x, indices_or_sections)

# TODO this doesn't exist for symbol.
def shape(x):
    return x.shape

def dtype(x):
    return x.dtype

def isinteger(x):
    return x.dtype in [np.int, np.int8, np.int16, np.int32, np.int64]

#unique = th.unique

def gather_row(data, row_index):
    return data[row_index,]

scatter_row = mx.nd.contrib.index_copy

def broadcast_to(x, to_array):
    return x + F.zeros_like(to_array)

squeeze = F.squeeze
unsqueeze = F.expand_dims
# TODO this doesn't exist for symbol.
reshape = F.reshape
ones = F.ones
zeros = F.zeros
arange = F.arange

def sort(x, dim=None, descending=False):
    if dim is None:
        dim = -1
    ascend = not descending
    # TODO this isn't an ideal implementation.
    val = F.sort(x, axis=dim, is_ascend=ascend)
    idx = F.argsort(x, axis=dim, is_ascend=ascend)
    idx = F.cast(idx, dtype='int64')
    return val, idx

def to_context(x, ctx):
    if ctx is None:
        return x
    elif ctx.device_type == TVMContext.STR2MASK['cuda']:
        return x.as_in_context(mx.gpu(ctx.device_id))
    elif ctx.device_type == TVMContext.STR2MASK['cpu']:
        return x.as_in_context(mx.cpu())
    else:
        raise RuntimeError('Invalid context', ctx)

def get_context(x):
    if x.context.device_type == 'cpu':
        return TVMContext(TVMContext.STR2MASK['cpu'], 0)
    else:
        return TVMContext(
                TVMContext.STR2MASK[x.context.device_type], x.context.device_id)

def _typestr(arr_dtype):
    return arr_dtype

def astvmarray(arr_data):
    """Return a TVMArray representation of the underlying data."""
    data = arr_data
    arr = TVMArray()
    shape = c_array(tvm_shape_index_t, tuple(data.shape))
    arr.data = ctypes.cast(data.data_ptr(), ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = TVMType(_typestr(data.dtype))
    arr.ndim = len(shape)
    arr.ctx = get_context(data)
    return arr

from mxnet.symbol.contrib import _cut_subgraph, _get_unique_subgraph_name, _get_graph_inputs
from mxnet.attribute import AttrScope
from mxnet.base import _as_list

def _create_prefix(prefix, name):
    return prefix + "_" + name + "_"

def _get_prefix(sym_name, gname):
    splits = sym_name.split(gname)
    if len(splits) == 1:
        return None
    else:
        return splits[0] + gname + "_"

def _remove_prefix(sym_name, gname):
    splits = sym_name.split(gname + "_")
    if len(splits) == 1:
        return sym_name
    else:
        return splits[1]

def _construct_subgraph(sym_out, name):
    if isinstance(sym_out, dict):
        sym_out = [sym_out[key] for key in sym_out]
    else:
        sym_out = _as_list(sym_out)
    g = mx.sym.Group(sym_out)

    flat_out = []
    all_input_names = g.list_inputs()
    output_names = {o.name for o in sym_out}
    for o in sym_out:
        if o.name in all_input_names or o.list_attr().get("__subgraph_name__", "") != name:
            flat_out.append(mx.sym.op.identity(o))
        else:
            flat_out.append(o)
    return mx.sym.Group(flat_out)

# In this case, all inputs are from external sources.
def _add_msg_inputs(g, inputs, vertex_frame, edge_frame, name):
    cut_syms = _cut_subgraph(g)
    input_syms = _get_graph_inputs(g)
    cut_sym_map = {sym.list_outputs()[0]:sym for sym in cut_syms}
    in_sym_map = {sym.list_outputs()[0]:sym for sym in input_syms}
    index = []
    src_prefix = _create_prefix("src", name)
    dst_prefix = _create_prefix("dst", name)
    edge_prefix = _create_prefix("edge", name)
    for in1 in _get_graph_inputs(g):
        if _get_prefix(in1.name, name) in [src_prefix, dst_prefix]:
            inputs.append(vertex_frame[_remove_prefix(in1.name, name)])
        elif _get_prefix(in1.name, name) == edge_prefix:
            inputs.append(edge_frame[_remove_prefix(in1.name, name)])
        elif in1.name in cut_sym_map.keys():
            inputs.append(cut_sym_map[in1.name])
        else:
            inputs.append(in_sym_map[in1.name])
        index.append(len(inputs) - 1)
    return index

# In this case, some inputs are from external sources and some inputs
# are the output of the previous function.
def _add_inputs(g, inputs, vertex_frame, prev_out_syms, name):
    cut_syms = _cut_subgraph(g)
    input_syms = _get_graph_inputs(g)
    cut_sym_map = {sym.list_outputs()[0]:sym for sym in cut_syms}
    prev_out_map = {sym.list_outputs()[0]:sym for sym in prev_out_syms}
    in_sym_map = {sym.list_outputs()[0]:sym for sym in input_syms}
    index = []

    def _get_index(syms, name):
        for i, sym in enumerate(syms):
            if sym.name == name:
                return i
        return -1

    for in1 in _get_graph_inputs(g):
        if in1.name in prev_out_map.keys():
            index.append((_get_index(prev_out_syms, in1.name), False))
        elif in1.name in vertex_frame.keys():
            inputs.append(vertex_frame[in1.name])
            index.append((len(inputs) - 1, True))
        elif in1.name in cut_sym_map.keys():
            inputs.append(cut_sym_map[in1.name])
            index.append((len(inputs) - 1, True))
        else:
            inputs.append(in_sym_map[in1.name])
            index.append((len(inputs) - 1, True))
    return index

def _get_list(syms):
    if isinstance(syms, mx.sym.Symbol):
        return [syms]
    elif isinstance(syms, dict):
        return [syms[key] for key in syms]
    else:
        return syms

def _send_and_recv(uid, vid, eid, recv_vid,
                   vertex_frame, edge_frame,
                   message_func, reduce_func, apply_node_func,
                   name="sr"):
    # We need to construct symbols required by all of the functions.
    name = _get_unique_subgraph_name(name)
    with AttrScope(__subgraph_name__=name):
        # TODO we break the operation below into pieces.
        # TODO we need to add destination vertices in the message function.
        vertex_frame = {key:mx.sym.take(vertex_frame[key], uid) for key in vertex_frame}
        edge_frame = {key:mx_sym.take(edge_frame[key], eid) for key in edge_frame}
        msg_syms = message_func(vertex_frame, edge_frame)
        msg_g = _construct_subgraph(msg_syms, name)

        red_node_syms = {}
        for key in vertex_frame:
            red_node_syms[key] = mx.sym.var(key)
        # We need to construct a message array sliced from the output of the message function.
        if isinstance(msg_syms, dict):
            # We should create new symbols for the outputs of the message function.
            msg_syms = {key:mx.sym.var(msg_syms[key].name) for key in msg_syms}
        else:
            msg_syms = mx.sym.var(msg_syms.name)
        red_syms = reduce_func(red_node_syms, msg_syms)
        red_g = _construct_subgraph(red_syms, name)

        # construct the symbols for the outputs of the reduce function.
        if isinstance(red_syms, dict):
            red_syms = {key:mx.sym.var(red_syms[key].name) for key in red_syms}
        else:
            red_syms = mx.sym.var(red_syms[key].name)
        update_node_syms = {key:mx.sym.var(key) for key in vertex_frame}
        update_node_syms.update(red_syms)
        # TODO how to pass a dict to a hybrid forward?
        update_syms = apply_node_func(update_node_syms['accum'])
        update_g = _construct_subgraph(update_syms, name)

    inputs = []
    inputs.append(uid)
    inputs.append(vid)
    inputs.append(eid)
    inputs.append(recv_vid)

    msg_index = _add_msg_inputs(msg_g, inputs, vertex_frame, edge_frame, name)
    red_index = _add_inputs(red_g, inputs, vertex_frame, _get_list(msg_syms), name)
    print(red_g.list_inputs())
    print(red_index)
    update_index = _add_inputs(update_g, inputs, vertex_frame, _get_list(red_syms), name)
    print(update_g.list_inputs())
    print(update_index)
    num_msg_out = 0
    for i in range(len(red_index)):
        if not red_index[i][1]:
            t = (red_index[i][0] + len(inputs), True)
            red_index[i] = t
            num_msg_out += 1
    for i in range(len(update_index)):
        if not update_index[i][1]:
            t = (update_index[i][0] + len(inputs) + num_msg_out, True)
            update_index[i] = t
    red_index = [i[0] for i in red_index]
    update_index = [i[0] for i in update_index]

    ret = mx.sym._internal._send_and_recv(msg_g, red_g, update_g, *inputs, msg_index=msg_index,
                                          red_index=red_index, update_index=update_index,
                                          num_outputs=len(update_g.list_outputs()))
