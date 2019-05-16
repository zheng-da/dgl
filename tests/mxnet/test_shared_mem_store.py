import dgl
import sys
import random
import time
import numpy as np
from multiprocessing import Process
from scipy import sparse as spsp
import mxnet as mx
import backend as F
import unittest
import dgl.function as fn

num_nodes = 100
num_edges = int(num_nodes * num_nodes * 0.1)
rand_port = random.randint(5000, 8000)
print('run graph store with port ' + str(rand_port), file=sys.stderr)

def check_array_shared_memory(g, worker_id, arrays):
    idx = dgl.utils.toindex(np.array([5]))
    if worker_id == 0:
        for i, arr in enumerate(arrays):
            arr.update(idx, mx.nd.ones(shape=(1, 10)) * i, inplace=True)
        g._sync_barrier()
    else:
        g._sync_barrier()
        for i, arr in enumerate(arrays):
            assert np.all(arr[0][idx].asnumpy() == i)

def check_init_func(worker_id, graph_name):
    time.sleep(3)
    print("worker starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_from_store(graph_name, "shared_mem", port=rand_port)
    # Verify the graph structure loaded from the shared memory.
    src, dst = g.all_edges()
    coo = csr.tocoo()
    assert F.array_equal(dst, F.tensor(coo.row))
    assert F.array_equal(src, F.tensor(coo.col))
    assert F.array_equal(g.ndata['feat'][0], F.tensor(np.arange(10), dtype=np.float32))
    assert F.array_equal(g.edata['feat'][0], F.tensor(np.arange(10), dtype=np.float32))
    g.init_ndata('test4', (g.number_of_nodes(), 10), 'float32')
    g.init_edata('test4', (g.number_of_edges(), 10), 'float32')
    g._sync_barrier()
    check_array_shared_memory(g, worker_id,
                              [g._node_frame._frame['test4'], g._edge_frame._frame['test4']])
    g.destroy()

def server_func(num_workers, graph_name):
    print("server starts")
    np.random.seed(0)
    csr = (spsp.random(num_nodes, num_nodes, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_store_server(csr, graph_name, "shared_mem", num_workers,
                                                          False, edge_dir="in", port=rand_port)
    assert num_nodes == g._graph.number_of_nodes()
    assert num_edges == g._graph.number_of_edges()
    g.ndata['feat'] = mx.nd.arange(num_nodes * 10).reshape((num_nodes, 10))
    g.edata['feat'] = mx.nd.arange(num_edges * 10).reshape((num_edges, 10))
    g.run()

def test_test_init():
    serv_p = Process(target=server_func, args=(2, 'test_graph1'))
    work_p1 = Process(target=check_init_func, args=(0, 'test_graph1'))
    work_p2 = Process(target=check_init_func, args=(1, 'test_graph1'))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()


def check_update_all_func(worker_id, graph_name):
    time.sleep(3)
    print("worker starts")
    g = dgl.contrib.graph_store.create_graph_from_store(graph_name, "shared_mem", port=rand_port)
    g._sync_barrier()
    g.dist_update_all(fn.copy_src(src='feat', out='m'),
                      fn.sum(msg='m', out='preprocess'))
    adj = g.adjacency_matrix()
    tmp = mx.nd.dot(adj, g.ndata['feat'])
    assert np.all((g.ndata['preprocess'] == tmp).asnumpy())
    g._sync_barrier()
    check_array_shared_memory(g, worker_id, [g.ndata['preprocess']])
    g.destroy()

def test_update_all():
    serv_p = Process(target=server_func, args=(2, 'test_graph3'))
    work_p1 = Process(target=check_update_all_func, args=(0, 'test_graph3'))
    work_p2 = Process(target=check_update_all_func, args=(1, 'test_graph3'))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    serv_p.join()
    work_p1.join()
    work_p2.join()

def server_func1(num_workers):
    print("server starts")
    np.random.seed(0)
    csr = (spsp.random(10, 10, density=0.1, format='csr') != 0).astype(np.int64)

    g = dgl.contrib.graph_store.create_graph_store_server(csr, "cgraph", "shared_mem",
                                                          num_workers, False, edge_dir="in")
    g.ndata['feat'] = mx.nd.zeros(shape=(g._graph.number_of_nodes(), 10))
    g.run()

def read_func(worker_id):
    time.sleep(3)
    print('reader runs: ' + str(worker_id))
    g = dgl.contrib.graph_store.create_graph_from_store("cgraph", "shared_mem")
    for i in range(100000):
        idx = dgl.utils.toindex(np.random.randint(0, g.number_of_nodes(), size=10))
        rows = g._node_frame._frame['feat'][idx].asnumpy()
        for j in range(len(rows)):
            assert np.all(rows[j] == rows[j][0])
    print('reader ends: ' + str(worker_id))

def write_func(worker_id):
    time.sleep(3)
    print('writer runs: ' + str(worker_id))
    g = dgl.contrib.graph_store.create_graph_from_store("cgraph", "shared_mem")
    rows = mx.nd.ones(shape=(10, 10))
    for i in range(100000):
        idx = dgl.utils.toindex(np.random.randint(0, g.number_of_nodes(), size=10))
        rows = rows + random.randint(0, 100)
        g._node_frame._frame['feat'].update(idx, rows, inplace=True)
    print('writer ends: ' + str(worker_id))

def test_concurrent():
    serv_p = Process(target=server_func1, args=(4,))
    work_p1 = Process(target=read_func, args=(0,))
    work_p2 = Process(target=read_func, args=(1,))
    work_p3 = Process(target=write_func, args=(2,))
    work_p4 = Process(target=write_func, args=(3,))
    serv_p.start()
    work_p1.start()
    work_p2.start()
    work_p3.start()
    work_p4.start()
    work_p1.join()
    work_p2.join()
    work_p3.join()
    work_p4.join()
    print('wait for s')
    serv_p.join()
    print('end')

if __name__ == '__main__':
    test_test_init()
    test_update_all()
    test_concurrent()