/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor.cc
 * \brief Definition of neighborhood-based sampler APIs.
 */

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/sampling/neighbor.h>
#include "../../../c_api_common.h"
#include "../../unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

HeteroSubgraph SampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace) {

  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes())
    << "Number of fanout values must match the number of edge types.";
  CHECK_EQ(prob.size(), hg->NumEdgeTypes())
    << "Number of probability tensors must match the number of edge types.";

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = aten::NullArray();
    } else if (fanouts[etype] == -1) {
      const auto &earr = (dir == EdgeDir::kOut) ?
        hg->OutEdges(etype, nodes_ntype) :
        hg->InEdges(etype, nodes_ntype);
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges[etype] = earr.id;
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? csr_code : csc_code;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseSampling(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, fanouts[etype], prob[etype], replace));
          } else {
            sampled_coo = aten::COORowWiseSampling(
              hg->GetCOOMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSRMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseSampling(
            hg->GetCSCMatrix(etype), nodes_ntype, fanouts[etype], prob[etype], replace);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
        sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph SampleNeighborsTopk(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& k,
    EdgeDir dir,
    const std::vector<FloatArray>& weight,
    bool ascending) {
  // sanity check
  CHECK_EQ(nodes.size(), hg->NumVertexTypes())
    << "Number of node ID tensors must match the number of node types.";
  CHECK_EQ(k.size(), hg->NumEdgeTypes())
    << "Number of k values must match the number of edge types.";
  CHECK_EQ(weight.size(), hg->NumEdgeTypes())
    << "Number of weight tensors must match the number of edge types.";

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0 || k[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        hg->DataType(), hg->Context());
      induced_edges[etype] = aten::NullArray();
    } else if (k[etype] == -1) {
      const auto &earr = (dir == EdgeDir::kOut) ?
        hg->OutEdges(etype, nodes_ntype) :
        hg->InEdges(etype, nodes_ntype);
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges[etype] = earr.id;
    } else {
      // sample from one relation graph
      auto req_fmt = (dir == EdgeDir::kOut)? csr_code : csc_code;
      auto avail_fmt = hg->SelectFormat(etype, req_fmt);
      COOMatrix sampled_coo;
      switch (avail_fmt) {
        case SparseFormat::kCOO:
          if (dir == EdgeDir::kIn) {
            sampled_coo = aten::COOTranspose(aten::COORowWiseTopk(
              aten::COOTranspose(hg->GetCOOMatrix(etype)),
              nodes_ntype, k[etype], weight[etype], ascending));
          } else {
            sampled_coo = aten::COORowWiseTopk(
              hg->GetCOOMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          }
          break;
        case SparseFormat::kCSR:
          CHECK(dir == EdgeDir::kOut) << "Cannot sample out edges on CSC matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSRMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          break;
        case SparseFormat::kCSC:
          CHECK(dir == EdgeDir::kIn) << "Cannot sample in edges on CSR matrix.";
          sampled_coo = aten::CSRRowWiseTopk(
            hg->GetCSCMatrix(etype), nodes_ntype, k[etype], weight[etype], ascending);
          sampled_coo = aten::COOTranspose(sampled_coo);
          break;
        default:
          LOG(FATAL) << "Unsupported sparse format.";
      }
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows, sampled_coo.num_cols,
        sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }

  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    IdArray fanouts_array = args[2];
    const auto& fanouts = fanouts_array.ToVector<int64_t>();
    const std::string dir_str = args[3];
    const auto& prob = ListValueToVector<FloatArray>(args[4]);
    const bool replace = args[5];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
    EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighbors(
        hg.sptr(), nodes, fanouts, dir, prob, replace);

    *rv = HeteroSubgraphRef(subg);
  });

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighborsTopk")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    IdArray k_array = args[2];
    const auto& k = k_array.ToVector<int64_t>();
    const std::string dir_str = args[3];
    const auto& weight = ListValueToVector<FloatArray>(args[4]);
    const bool ascending = args[5];

    CHECK(dir_str == "in" || dir_str == "out")
      << "Invalid edge direction. Must be \"in\" or \"out\".";
      EdgeDir dir = (dir_str == "in")? EdgeDir::kIn : EdgeDir::kOut;

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighborsTopk(
        hg.sptr(), nodes, k, dir, weight, ascending);

    *rv = HeteroGraphRef(subg);
  });

template<class T>
std::vector<T> &get_vector() {
  static thread_local std::vector<T> vec;
  return vec;
}

template<class T>
class hashtable
{
  std::vector<T> &hmap;
  T mask;
 public:
  hashtable(long n): hmap(get_vector<T>()) {
    long new_size;
    for (new_size=1; new_size<3*n; new_size*=2);
    mask = new_size-1;
    if (new_size > (long) hmap.size()) {
      hmap.resize(new_size, -1);
    } else {
      for (long i = 0; i < new_size; i++)
        hmap[i] = -1;
    }
  }

  ~hashtable() {
  }

  bool insert(T k) {
    long j;
    for (j=(k&mask); hmap[j]!=-1 && hmap[j]!=k; j=((j+1)&mask));
    if (hmap[j] == -1) {
      hmap[j] = k;
      return true;
    } else {
      return false;
    }
  }

  bool contain(T k) const {
    long j;
    for (j=(k&mask); hmap[j]!=-1 && hmap[j]!=k; j=((j+1)&mask));
    return hmap[j] != -1;
  }
};

template<class T>
class BufferData {
  hashtable<T> map;
  std::vector<T> arr;
 public:
  BufferData(const T*raw_arr, int64_t len): map(len) {
    for (int64_t i = 0; i < len; i++) {
      map.insert(raw_arr[i]);
      arr.push_back(raw_arr[i]);
      std::sort(arr.begin(), arr.end());
    }
  }

  std::pair<std::vector<T>, std::vector<int64_t>> lookup(const T*raw_arr, int64_t len) const {
    std::vector<T> res;
    std::vector<int64_t> offs;
    // If this input vector is large, we can scan the input vector with the
    // sorted buffer to check the overlap.
    if (len >= res.size() / 2) {
      res.resize(std::max((size_t) len, res.size()));
      auto it = std::set_intersection(raw_arr, raw_arr + len, arr.begin(), arr.end(), res.begin());
      res.resize(it - res.begin());
      // TODO we need to get offsets.
    }
    else {
      // Otherwise, we iterate the elements in the input vector.
      for (int64_t i = 0; i < len; i++) {
        if (map.contain(raw_arr[i])) {
          res.push_back(raw_arr[i]);
          offs.push_back(i);
        }
      }
    }

    return std::pair<std::vector<T>, std::vector<int64_t>>(res, offs);
  }
};

std::shared_ptr<BufferData<dgl_id_t>> buf;

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLCreateSampleBuffer")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    IdArray buf_arr = args[0];
    int64_t len = buf_arr->shape[0];
    const dgl_id_t *buf_data = static_cast<dgl_id_t*>(buf_arr->data);
    buf = std::shared_ptr<BufferData<dgl_id_t>>(new BufferData<dgl_id_t>(buf_data, len));
  });

HeteroSubgraph SampleNeighborsWithBuffer(
    const HeteroGraphPtr hg,
    IdArray seed_nodes) {
  CHECK_EQ(hg->NumEdgeTypes(), 1);
  const dgl_id_t *seed_data = static_cast<dgl_id_t *>(seed_nodes->data);
  int64_t num_seeds = seed_nodes->shape[0];

  // Sample edges.
  std::vector<dgl_id_t> src_ids, dst_ids, eids;
  for (int64_t i = 0; i < num_seeds; i++) {
    dgl_id_t seed = seed_data[i];
    auto neigh_it = hg->PredVec(0, seed);
    auto edge_it = hg->InEdgeVec(0, seed);
    auto res = buf->lookup(neigh_it.begin(), neigh_it.end() - neigh_it.begin());
    auto &sampled_ids = res.first;
    auto &offs = res.second;
    assert(offs.size() == sampled_ids.size());
    for (size_t j = 0; j < sampled_ids.size(); j++) {
      src_ids.push_back(seed);
      dst_ids.push_back(sampled_ids[j]);
      eids.push_back(edge_it.begin()[offs[j]]);
    }
  }

  // Move data to IdArrays
  int64_t num_sampled = src_ids.size();
  IdArray src = IdArray::Empty({num_sampled},
                               DLDataType{kDLInt, sizeof(dgl_id_t) * 8, 1}, DLContext{kDLCPU, 0});
  memcpy(src->data, src_ids.data(), num_sampled * sizeof(dgl_id_t));
  IdArray dst = IdArray::Empty({num_sampled},
                               DLDataType{kDLInt, sizeof(dgl_id_t) * 8, 1}, DLContext{kDLCPU, 0});
  memcpy(dst->data, dst_ids.data(), num_sampled * sizeof(dgl_id_t));
  IdArray eid = IdArray::Empty({num_sampled},
                               DLDataType{kDLInt, sizeof(dgl_id_t) * 8, 1}, DLContext{kDLCPU, 0});
  memcpy(eid->data, eids.data(), num_sampled * sizeof(dgl_id_t));

  // Construct the subgraph.
  auto subg = UnitGraph::CreateFromCOO(hg->GetRelationGraph(0)->NumVertexTypes(),
                                       hg->NumVertices(0),
                                       hg->NumVertices(0),
                                       src, dst);
  HeteroSubgraph ret;
  std::vector<HeteroGraphPtr> subgs(1);
  subgs[0] = subg;
  ret.graph = CreateHeteroGraph(hg->meta_graph(), subgs, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  std::vector<IdArray> induced_edges(1);
  induced_edges[0] = eid;
  ret.induced_edges = induced_edges;
  return ret;
}

DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleWithBuffer")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    IdArray seed_nodes = args[1];

    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = sampling::SampleNeighborsWithBuffer(hg.sptr(), seed_nodes);

    *rv = HeteroGraphRef(subg);
  });

}  // namespace sampling
}  // namespace dgl
