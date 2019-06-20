import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import backend as F
from dgl import DGLError

D = 5

# graph generation: a random bipartite graph with 10 left nodes and 11 right nodes.
#  and 21 edges.
#  - has self loop
#  - no multi edge
def edge_pair_input(sort=False):
    if sort:
        src = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 7, 9, 9]
        dst = [4, 6, 9, 3, 5, 3, 7, 5, 8, 1, 3, 4, 9, 1, 9, 6, 2, 8, 9, 2, 10]
        return src, dst
    else:
        src = [0, 0, 4, 5, 0, 4, 7, 4, 4, 3, 2, 7, 7, 5, 3, 2, 1, 9, 6, 1, 9]
        dst = [9, 6, 3, 9, 4, 4, 9, 9, 1, 8, 3, 2, 8, 1, 5, 7, 3, 2, 6, 5, 10]
        return src, dst

def gen_from_edgelist(directed):
    src, dst = edge_pair_input()
    num_typed_nodes = {'src': max(src) + 1, 'dst': max(dst) + 1}
    src = np.array(src, np.int64)
    dst = np.array(dst, np.int64)
    metagraph = nx.MultiGraph([('src', 'dst', 'e'), ('dst', 'src', 'e')])
    if directed:
        g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                                  {('src', 'dst', 'e'): (src, dst)},
                                  readonly=True)
    else:
        g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                                  {('src', 'dst', 'e'): (src, dst),
                                   ('dst', 'src', 'e'): (dst, src)},
                                  readonly=True)
    return g

def scipy_coo_input():
    src, dst = edge_pair_input()
    return sp.coo_matrix((np.ones((len(src),)), (src, dst)), shape=(10,11))

def scipy_csr_input():
    src, dst = edge_pair_input()
    csr = sp.coo_matrix((np.ones((len(src),)), (src, dst)), shape=(10,11)).tocsr()
    csr.sort_indices()
    coo = csr.tocoo()
    # src = [0 0 0 1 1 2 2 3 3 4 4 4 4 5 5 6 7 7 7 9]
    # dst = [4 6 9 3 5 3 7 5 8 1 3 4 9 1 9 6 2 8 9 2]
    return csr

#def gen_by_mutation():
#    g = dgl.DGLGraph()
#    src, dst = edge_pair_input()
#    g.add_nodes(10)
#    g.add_edges(src, dst)
#    return g

def gen_from_data(data, readonly):
    num_typed_nodes = {'src': data.shape[0], 'dst': data.shape[1]}
    metagraph = nx.MultiGraph([('src', 'dst', 'e')])
    g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                              {('src', 'dst', 'e'): data}, readonly=readonly)
    return g

def test_query():
    def _test_one1(g):
        assert g['src'].number_of_nodes() == 10
        assert g['dst'].number_of_nodes() == 11
        assert g.number_of_edges() == 21
        assert not g.is_multigraph

        for i in range(10):
            assert g['src'].has_node(i)
            assert i in g['src']
        for i in range(11):
            assert g['dst'].has_node(i)
            assert i in g['dst']
        assert not g['src'].has_node(11)
        assert not g['dst'].has_nodes(12)
        assert not g['src'].has_node(-1)
        assert not g['dst'].has_node(-1)
        assert not -1 in g['src']
        assert F.allclose(g['src'].has_nodes([-1,0,2,10,11]), F.tensor([0,1,1,0,0]))

        src, dst = edge_pair_input()
        for u, v in zip(src, dst):
            assert g.has_edge_between(u, v)
        assert not g.has_edge_between(0, 0)
        assert F.allclose(g.has_edges_between([0, 0, 3], [0, 9, 8]), F.tensor([0,1,1]))
        assert set(F.asnumpy(g.predecessors(9))) == set([0,5,7,4])
        assert set(F.asnumpy(g.successors(2))) == set([7,3])

        assert g.edge_id(4,4) == 5
        assert F.allclose(g.edge_ids([4,0], [4,9]), F.tensor([5,0]))

        src, dst = g.find_edges([3, 6, 5])
        assert F.allclose(src, F.tensor([5, 7, 4]))
        assert F.allclose(dst, F.tensor([9, 9, 4]))

        src, dst, eid = g.in_edges(9, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7)])
        src, dst, eid = g.in_edges([9,0,8], form='all')  # test node#0 has no in edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7),(3,8,9),(7,8,12)])

        src, dst, eid = g.out_edges(0, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4)])
        src, dst, eid = g.out_edges([0,4,8], form='all')  # test node#8 has no out edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4),(4,3,2),(4,4,5),(4,9,7),(4,1,8)])

        src, dst, eid = g.edges('all', 'eid')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(eid)) == list(range(21))

        src, dst, eid = g.edges('all', 'srcdst')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(src)) == sorted(list(F.asnumpy(src)))

        assert g.in_degree(0) == 0
        assert g.in_degree(9) == 4
        assert F.allclose(g.in_degrees([0, 9]), F.tensor([0, 4]))
        assert g.out_degree(8) == 0
        assert g.out_degree(9) == 2
        assert F.allclose(g.out_degrees([8, 9]), F.tensor([0, 2]))

        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('src', 'dst', 'e'))),
                              scipy_coo_input().toarray().T)
        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('src', 'dst', 'e'), transpose=True)),
                              scipy_coo_input().toarray())

    def _test_one2(g):
        assert g['src'].number_of_nodes() == 10
        assert g['dst'].number_of_nodes() == 11
        assert g.number_of_edges() == 21
        assert not g.is_multigraph

        for i in range(10):
            assert g['src'].has_node(i)
            assert i in g['src']
        for i in range(11):
            assert g['dst'].has_node(i)
            assert i in g['dst']
        assert not g['src'].has_node(11)
        assert not g['dst'].has_nodes(12)
        assert not g['src'].has_node(-1)
        assert not g['dst'].has_node(-1)
        assert not -1 in g['src']
        assert F.allclose(g['src'].has_nodes([-1,0,2,10,11]), F.tensor([0,1,1,0,0]))

        src, dst = edge_pair_input()
        for u, v in zip(dst, src):
            assert g.has_edge_between(u, v)
        assert not g.has_edge_between(0, 0)
        assert F.allclose(g.has_edges_between([0, 9, 8], [0, 0, 3]), F.tensor([0,1,1]))
        assert set(F.asnumpy(g.successors(9))) == set([0,5,7,4])
        assert set(F.asnumpy(g.predecessors(2))) == set([7,3])

        assert g.edge_id(4,4) == 5
        assert F.allclose(g.edge_ids([4,9], [4,0]), F.tensor([5,0]))

        src, dst = g.find_edges([3, 6, 5])
        assert F.allclose(dst, F.tensor([5, 7, 4]))
        assert F.allclose(src, F.tensor([9, 9, 4]))

        src, dst, eid = g.out_edges(9, form='all')
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7)])
        src, dst, eid = g.out_edges([9,0,8], form='all')  # test node#0 has no in edges
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7),(3,8,9),(7,8,12)])

        src, dst, eid = g.in_edges(0, form='all')
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4)])
        src, dst, eid = g.in_edges([0,4,8], form='all')  # test node#8 has no out edges
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4),(4,3,2),(4,4,5),(4,9,7),(4,1,8)])

        src, dst, eid = g.edges('all', 'eid')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(eid)) == list(range(21))

        src, dst, eid = g.edges('all', 'srcdst')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(dst), F.asnumpy(src), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(src)) == sorted(list(F.asnumpy(src)))

        assert g.out_degree(0) == 0
        assert g.out_degree(9) == 4
        assert F.allclose(g.out_degrees([0, 9]), F.tensor([0, 4]))
        assert g.in_degree(8) == 0
        assert g.in_degree(9) == 2
        assert F.allclose(g.in_degrees([8, 9]), F.tensor([0, 2]))

        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('dst', 'src', 'e'))),
                              scipy_coo_input().toarray())
        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('dst', 'src', 'e'), transpose=True)),
                              scipy_coo_input().toarray().T)

    def _test1(g):
        # test twice to see whether the cached format works or not
        _test_one1(g)
        _test_one1(g)

    def _test2(g):
        # test twice to see whether the cached format works or not
        _test_one2(g)
        _test_one2(g)

    def _test_csr_one(g):
        assert g['src'].number_of_nodes() == 10
        assert g['dst'].number_of_nodes() == 11
        assert g.number_of_edges() == 21
        assert len(g['src']) == 10
        assert len(g['dst']) == 11
        assert not g.is_multigraph

        for i in range(10):
            assert g['src'].has_node(i)
            assert i in g['src']
        for i in range(11):
            assert g['dst'].has_node(i)
            assert i in g['dst']
        assert not g['src'].has_node(11)
        assert not g['dst'].has_node(12)
        assert not g['src'].has_node(-1)
        assert not -1 in g['src']
        assert F.allclose(g['src'].has_nodes([-1,0,2,10,11]), F.tensor([0,1,1,0,0]))

        src, dst = edge_pair_input(sort=True)
        for u, v in zip(src, dst):
            assert g.has_edge_between(u, v)
        assert not g.has_edge_between(0, 0)
        assert F.allclose(g.has_edges_between([0, 0, 3], [0, 9, 8]), F.tensor([0,1,1]))
        assert set(F.asnumpy(g.predecessors(9))) == set([0,5,7,4])
        assert set(F.asnumpy(g.successors(2))) == set([7,3])

        # src = [0 0 0 1 1 2 2 3 3 4 4 4 4 5 5 6 7 7 7 9]
        # dst = [4 6 9 3 5 3 7 5 8 1 3 4 9 1 9 6 2 8 9 2]
        # eid = [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
        assert g.edge_id(4,4) == 11
        assert F.allclose(g.edge_ids([4,0], [4,9]), F.tensor([11,2]))

        src, dst = g.find_edges([3, 6, 5])
        assert F.allclose(src, F.tensor([1, 2, 2]))
        assert F.allclose(dst, F.tensor([3, 7, 3]))

        src, dst, eid = g.in_edges(9, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(5,9,14),(7,9,18),(4,9,12)])
        src, dst, eid = g.in_edges([9,0,8], form='all')  # test node#0 has no in edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(5,9,14),(7,9,18),(4,9,12),(3,8,8),(7,8,17)])

        src, dst, eid = g.out_edges(0, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(0,6,1),(0,4,0)])
        src, dst, eid = g.out_edges([0,4,8], form='all')  # test node#8 has no out edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(0,6,1),(0,4,0),(4,3,10),(4,4,11),(4,9,12),(4,1,9)])

        src, dst, eid = g.edges('all', 'eid')
        t_src, t_dst = edge_pair_input(sort=True)
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(eid)) == list(range(21))

        src, dst, eid = g.edges('all', 'srcdst')
        t_src, t_dst = edge_pair_input(sort=True)
        t_tup = list(zip(t_src, t_dst, list(range(21))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(src)) == sorted(list(F.asnumpy(src)))

        assert g.in_degree(0) == 0
        assert g.in_degree(9) == 4
        assert F.allclose(g.in_degrees([0, 9]), F.tensor([0, 4]))
        assert g.out_degree(8) == 0
        assert g.out_degree(9) == 2
        assert F.allclose(g.out_degrees([8, 9]), F.tensor([0, 2]))

        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('src', 'dst', 'e'))),
                              scipy_coo_input().toarray().T)
        assert np.array_equal(F.sparse_to_numpy(g.adjacency_matrix(('src', 'dst', 'e'), transpose=True)),
                              scipy_coo_input().toarray())

    def _test_csr(g):
        # test twice to see whether the cached format works or not
        _test_csr_one(g)
        _test_csr_one(g)

    _test1(gen_from_edgelist(True))
    g = gen_from_edgelist(False)
    _test1(g['src', 'dst', 'e'])
    _test2(g['dst', 'src', 'e'])
    #_test(gen_from_data(scipy_coo_input(), False))
    _test1(gen_from_data(scipy_coo_input(), True))

    #_test_csr(gen_from_data(scipy_csr_input(), False))
    _test_csr(gen_from_data(scipy_csr_input(), True))

def test_mutation():
    g = dgl.DGLGraph()
    # test add nodes with data
    g.add_nodes(5)
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    ans = F.cat([F.zeros((5, 2)), F.ones((5, 2))], 0)
    assert F.allclose(ans, g.ndata['h'])
    g.ndata['w'] = 2 * F.ones((10, 2))
    assert F.allclose(2 * F.ones((10, 2)), g.ndata['w'])
    # test add edges with data
    g.add_edges([2, 3], [3, 4])
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    ans = F.cat([F.zeros((2, 2)), F.ones((2, 2))], 0)
    assert F.allclose(ans, g.edata['m'])
    # test clear and add again
    g.clear()
    g.add_nodes(5)
    g.ndata['h'] = 3 * F.ones((5, 2))
    assert F.allclose(3 * F.ones((5, 2)), g.ndata['h'])
    g.init_ndata('h1', (g.number_of_nodes(), 3), 'float32')
    assert F.allclose(F.zeros((g.number_of_nodes(), 3)), g.ndata['h1'])
    g.init_edata('h2', (g.number_of_edges(), 3), 'float32')
    assert F.allclose(F.zeros((g.number_of_edges(), 3)), g.edata['h2'])

def test_scipy_adjmat():
    g = gen_from_edgelist(False)
    coo = scipy_coo_input()
    coo_t = coo.transpose()

    adj_0 = g.adjacency_matrix_scipy(('src', 'dst', 'e'))
    adj_1 = g.adjacency_matrix_scipy(('src', 'dst', 'e'), fmt='coo')
    assert np.array_equal(coo_t.row, adj_1.row)
    assert np.array_equal(coo_t.col, adj_1.col)
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

    adj_0 = g.adjacency_matrix_scipy(('dst', 'src', 'e'))
    adj_1 = g.adjacency_matrix_scipy(('dst', 'src', 'e'), fmt='coo')
    assert np.array_equal(coo.row, adj_1.row)
    assert np.array_equal(coo.col, adj_1.col)
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

    adj_t0 = g.adjacency_matrix_scipy(('src', 'dst', 'e'), transpose=True)
    adj_t_1 = g.adjacency_matrix_scipy(('src', 'dst', 'e'), transpose=True, fmt='coo')
    assert np.array_equal(coo.row, adj_t_1.row)
    assert np.array_equal(coo.col, adj_t_1.col)
    assert np.array_equal(adj_t0.toarray(), adj_t_1.toarray())

    adj_t0 = g.adjacency_matrix_scipy(('dst', 'src', 'e'), transpose=True)
    adj_t_1 = g.adjacency_matrix_scipy(('dst', 'src', 'e'), transpose=True, fmt='coo')
    assert np.array_equal(coo_t.row, adj_t_1.row)
    assert np.array_equal(coo_t.col, adj_t_1.col)
    assert np.array_equal(adj_t0.toarray(), adj_t_1.toarray())

def test_incmat():
    src = [0, 0, 0, 2, 1]
    dst = [1, 2, 3, 3, 1]
    num_typed_nodes = {'src': max(src) + 1, 'dst': max(dst) + 1}
    src = np.array(src, np.int64)
    dst = np.array(dst, np.int64)
    metagraph = nx.MultiGraph([('src', 'dst', 'e')])
    g = dgl.DGLBipartiteGraph(metagraph, num_typed_nodes,
                              {('src', 'dst', 'e'): (src, dst)}, readonly=True)
    inc_in = F.sparse_to_numpy(g.incidence_matrix(('src', 'dst', 'e'), 'in'))
    inc_out = F.sparse_to_numpy(g.incidence_matrix(('src', 'dst', 'e'), 'out'))
    assert np.allclose(
            inc_in,
            np.array([[0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 1.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 1., 0.]]))
    assert np.allclose(
            inc_out,
            np.array([[1., 1., 1., 0., 0.],
                      [0., 0., 0., 0., 1.],
                      [0., 0., 0., 1., 0.]]))

def test_readonly():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([0, 1, 2, 3], [1, 2, 3, 4])
    g.ndata['x'] = F.zeros((5, 3))
    g.edata['x'] = F.zeros((4, 4))

    g.readonly(False)
    assert g._graph.is_readonly() == False
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    g.readonly()
    assert g._graph.is_readonly() == True 
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(5)
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly()
    assert g._graph.is_readonly() == True 
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(5)
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly(False)
    assert g._graph.is_readonly() == False
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(10)
        g.add_edges([4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert not fail
        assert g.number_of_nodes() == 15
        assert F.shape(g.ndata['x']) == (15, 3)
        assert g.number_of_edges() == 14
        assert F.shape(g.edata['x']) == (14, 4)

def test_find_edges():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(range(9), range(1, 10))
    e = g.find_edges([1, 3, 2, 4])
    assert e[0][0] == 1 and e[0][1] == 3 and e[0][2] == 2 and e[0][3] == 4
    assert e[1][0] == 2 and e[1][1] == 4 and e[1][2] == 3 and e[1][3] == 5

    try:
        g.find_edges([10])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly()
    e = g.find_edges([1, 3, 2, 4])
    assert e[0][0] == 1 and e[0][1] == 3 and e[0][2] == 2 and e[0][3] == 4
    assert e[1][0] == 2 and e[1][1] == 4 and e[1][2] == 3 and e[1][3] == 5

    try:
        g.find_edges([10])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

def check_apply_nodes(g, ntype):
    def _upd(nodes):
        return {'h' : nodes.data['h'] * 2}
    g[ntype].ndata['h'] = F.randn((g[ntype].number_of_nodes(), D))
    g[ntype].register_apply_node_func(_upd)
    old = g[ntype].ndata['h']
    g[ntype].apply_nodes()
    assert F.allclose(old * 2, g[ntype].ndata['h'])
    u = F.tensor([0, 3, 4, 6])
    g[ntype].apply_nodes(lambda nodes : {'h' : nodes.data['h'] * 0.}, u)
    assert F.allclose(F.gather_row(g[ntype].ndata['h'], u), F.zeros((4, D)))

def test_apply_nodes():
    g = gen_from_edgelist(False)
    check_apply_nodes(g, 'src')
    check_apply_nodes(g, 'dst')

def test_apply_edges():
    def _upd(edges):
        return {'w' : edges.data['w'] * 2}
    g = gen_from_edgelist(False)
    g['src', 'dst', 'e'].edata['w'] = F.randn((g['src', 'dst', 'e'].number_of_edges(), D))
    g['dst', 'src', 'e'].edata['w'] = F.randn((g['dst', 'src', 'e'].number_of_edges(), D))
    old1 = g['src', 'dst', 'e'].edata['w']
    old2 = g['dst', 'src', 'e'].edata['w']
    g.apply_edges({('src', 'dst', 'e'): _upd,
                   ('dst', 'src', 'e'): _upd})
    assert F.allclose(old1 * 2, g['src', 'dst', 'e'].edata['w'])
    assert F.allclose(old2 * 2, g['dst', 'src', 'e'].edata['w'])

    g.register_apply_edge_func({('src', 'dst', 'e'): _upd,
                                ('dst', 'src', 'e'): _upd})
    g.apply_edges()
    assert F.allclose(old1 * 4, g['src', 'dst', 'e'].edata['w'])
    assert F.allclose(old2 * 4, g['dst', 'src', 'e'].edata['w'])

    u = F.tensor([0, 0, 0, 4, 5, 6])
    v = F.tensor([4, 6, 9, 3, 9, 6])
    udf = lambda edges : {'w' : edges.data['w'] * 0.}
    g.apply_edges({('src', 'dst', 'e'): udf,
                   ('dst', 'src', 'e'): udf},
                  {('src', 'dst', 'e'): (u, v),
                   ('dst', 'src', 'e'): (v, u)})
    eid = g['src', 'dst', 'e'].edge_ids(u, v)
    assert F.allclose(F.gather_row(g['src', 'dst', 'e'].edata['w'], eid),
                      F.zeros((6, D)))

reduce_msg_shapes = set()

def message_func(edges):
    assert F.ndim(edges.src['h']) == 2
    assert F.shape(edges.src['h'])[1] == D
    return {'m' : edges.src['h']}

def reduce_func(nodes):
    msgs = nodes.mailbox['m']
    reduce_msg_shapes.add(tuple(msgs.shape))
    assert F.ndim(msgs) == 3
    assert F.shape(msgs)[2] == D
    return {'accum' : F.sum(msgs, 1)}

def apply_node_func(nodes):
    return {'res' : nodes.data['h'] + nodes.data['accum']}

def test_update_routines():
    g = gen_from_edgelist(False)
    g['src'].ndata['h'] = F.randn((g['src'].number_of_nodes(), D))
    g['dst'].ndata['h'] = F.randn((g['dst'].number_of_nodes(), D))
    adj = g.adjacency_matrix(('src', 'dst', 'e'))
    comp_res = F.spmm(adj, g['src'].ndata['h']) + g['dst'].ndata['h']

    # send_and_recv
    reduce_msg_shapes.clear()
    u = [0, 0, 0, 4, 5, 6]
    v = [4, 6, 9, 3, 9, 6]
    g.send_and_recv((u, v),
                    {('src', 'dst', 'e'): message_func},
                    {'dst': reduce_func},
                    {'dst': apply_node_func})
    assert(reduce_msg_shapes == {(2, 2, D), (2, 1, D)})

    reduce_msg_shapes.clear()
    u = [0, 0, 0, 4, 5, 6]
    v = [4, 6, 9, 3, 9, 6]
    g['src', 'dst', 'e'].send_and_recv((u, v),
                                       message_func,
                                       reduce_func,
                                       apply_node_func)
    assert(reduce_msg_shapes == {(2, 2, D), (2, 1, D)})

    reduce_msg_shapes.clear()
    try:
        g.send_and_recv([u, v])
        assert False
    except:
        pass

    # pull
    v = F.tensor([1, 2, 3, 9])
    reduce_msg_shapes.clear()
    g.pull(v,
           {('src', 'dst', 'e'): message_func},
           {'dst': reduce_func},
           {'dst': apply_node_func})
    assert(reduce_msg_shapes == {(2, 2, D), (1, 3, D), (1, 4, D)})
    reduce_msg_shapes.clear()
    F.allclose(g['dst'].ndata['res'][v], comp_res[v])

    g['src', 'dst', 'e'].pull(v,
                              message_func,
                              reduce_func,
                              apply_node_func)
    assert(reduce_msg_shapes == {(2, 2, D), (1, 3, D), (1, 4, D)})
    reduce_msg_shapes.clear()
    F.allclose(g['dst'].ndata['res'][v], comp_res[v])

    # push
    v = F.tensor([0, 1, 2, 3])
    reduce_msg_shapes.clear()
    g.push(v,
           {('src', 'dst', 'e'): message_func},
           {'dst': reduce_func},
           {'dst': apply_node_func})
    assert(reduce_msg_shapes == {(2, 2, D), (5, 1, D)})
    reduce_msg_shapes.clear()

    g['src', 'dst', 'e'].push(v,
                              message_func,
                              reduce_func,
                              apply_node_func)
    assert(reduce_msg_shapes == {(2, 2, D), (5, 1, D)})
    reduce_msg_shapes.clear()

    # update_all
    reduce_msg_shapes.clear()
    g.update_all({('src', 'dst', 'e'): message_func},
                 {'dst': reduce_func},
                 {'dst': apply_node_func})
    assert(reduce_msg_shapes == {(1, 3, D), (2, 1, D), (6, 2, D), (1, 4, D)})
    reduce_msg_shapes.clear()
    F.allclose(g['dst'].ndata['res'], comp_res)

    g['src', 'dst', 'e'].update_all(message_func,
                                    reduce_func,
                                    apply_node_func)
    assert(reduce_msg_shapes == {(1, 3, D), (2, 1, D), (6, 2, D), (1, 4, D)})
    reduce_msg_shapes.clear()
    F.allclose(g['dst'].ndata['res'], comp_res)

def test_filter():
    g = gen_from_edgelist(False)

    src_repr = F.zeros((g['src'].number_of_nodes(), 5))
    dst_repr = F.zeros((g['dst'].number_of_nodes(), 5))
    e_repr = F.zeros((g['src', 'dst', 'e'].number_of_edges(), 5))
    src_repr[[1, 3]] = 1
    dst_repr[[1, 5]] = 1
    e_repr[[1, 3]] = 1

    g['src'].ndata['a'] = src_repr
    g['dst'].ndata['a'] = dst_repr
    g['src', 'dst', 'e'].edata['a'] = e_repr

    def predicate(r):
        return F.max(r.data['a'], 1) > 0

    # full node filter
    n_idx = g.filter_nodes('src', predicate)
    assert set(F.zerocopy_to_numpy(n_idx)) == {1, 3}
    n_idx = g.filter_nodes('dst', predicate)
    assert set(F.zerocopy_to_numpy(n_idx)) == {1, 5}

    # partial node filter
    n_idx = g.filter_nodes('src', predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(n_idx)) == {1}

    # full edge filter
    e_idx = g.filter_edges(('src', 'dst', 'e'), predicate)
    assert set(F.zerocopy_to_numpy(e_idx)) == {1, 3}

    # partial edge filter
    e_idx = g.filter_edges(('src', 'dst', 'e'), predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(e_idx)) == {1}

if __name__ == '__main__':
    test_query()
    #test_mutation()
    test_scipy_adjmat()
    test_incmat()
    #test_readonly()
    #test_find_edges()
    test_apply_nodes()
    test_apply_edges()
    test_update_routines()
    test_filter()

