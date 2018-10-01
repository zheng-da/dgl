from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import scipy.sparse

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

#def scatter_row(data, row_index, value):
#    return data.index_copy(0, row_index, value)

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
