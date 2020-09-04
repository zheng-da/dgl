/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/to_bipartite.cc
 * \brief Convert a graph to a bipartite-structured graph.
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <vector>
#include <tuple>
// TODO(BarclayII): currently ToBlock depend on IdHashMap<IdType> implementation which
// only works on CPU.  Should fix later to make it device agnostic.
#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlock(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes, bool include_rhs_in_lhs) {
  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();
  std::vector<EdgeArray> edge_arrays(num_etypes);

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
    << "rhs_nodes not given for every node type";

  const std::vector<IdHashMap<IdType>> rhs_node_mappings(rhs_nodes.begin(), rhs_nodes.end());
  std::vector<IdHashMap<IdType>> lhs_node_mappings;

  if (include_rhs_in_lhs)
    lhs_node_mappings = rhs_node_mappings;  // copy
  else
    lhs_node_mappings.resize(num_ntypes);

  std::vector<int64_t> num_nodes_per_type;
  num_nodes_per_type.reserve(2 * num_ntypes);

  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      const EdgeArray& edges = graph->Edges(etype);
      lhs_node_mappings[srctype].Update(edges.src);
      edge_arrays[etype] = edges;
    }
  }

  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(lhs_node_mappings[ntype].Size());
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(rhs_node_mappings[ntype].Size());

  std::vector<HeteroGraphPtr> rel_graphs;
  std::vector<IdArray> induced_edges;
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const IdHashMap<IdType> &lhs_map = lhs_node_mappings[srctype];
    const IdHashMap<IdType> &rhs_map = rhs_node_mappings[dsttype];
    if (rhs_map.Size() == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_map.Size(), rhs_map.Size(),
          aten::NullArray(), aten::NullArray()));
      induced_edges.push_back(aten::NullArray());
    } else {
      IdArray new_src = lhs_map.Map(edge_arrays[etype].src, -1);
      IdArray new_dst = rhs_map.Map(edge_arrays[etype].dst, -1);
      // Check whether there are unmapped IDs and raise error.
      for (int64_t i = 0; i < new_dst->shape[0]; ++i)
        CHECK_NE(new_dst.Ptr<IdType>()[i], -1)
          << "Node " << edge_arrays[etype].dst.Ptr<IdType>()[i] << " does not exist"
          << " in `rhs_nodes`. Argument `rhs_nodes` must contain all the edge"
          << " destination nodes.";
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_map.Size(), rhs_map.Size(),
          new_src, new_dst));
      induced_edges.push_back(edge_arrays[etype].id);
    }
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);
  std::vector<IdArray> lhs_nodes;
  for (const IdHashMap<IdType> &lhs_map : lhs_node_mappings)
    lhs_nodes.push_back(lhs_map.Values());
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}

};  // namespace

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlock(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes, bool include_rhs_in_lhs) {
  std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>> ret;
  ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
    ret = ToBlock<IdType>(graph, rhs_nodes, include_rhs_in_lhs);
  });
  return ret;
}

std::pair<IdArray, IdArray>
CountingSort(IdArray keys, int num_keys) {
  std::pair<IdArray, IdArray> result;
  ATEN_ID_TYPE_SWITCH(keys->dtype, IdType, {
    IdArray count = aten::Full(0, num_keys, keys->dtype.bits, keys->ctx);
    IdType *count_data = count.Ptr<IdType>();
    std::vector<IdType> offset(num_keys);
    int64_t n = keys->shape[0];
    IdArray ret = aten::NewIdArray(n, keys->ctx, keys->dtype.bits);
    IdType *ret_data = ret.Ptr<IdType>();
    const IdType *keys_data = keys.Ptr<IdType>();

    for (int64_t i = 0; i < n; ++i)
      ++count_data[keys_data[i]];
    for (int64_t i = 0; i < num_keys; ++i)
      offset[i] = (i == 0) ? 0 : offset[i - 1] + count_data[i - 1];
    for (int64_t i = 0; i < n; ++i) {
      ret_data[offset[keys_data[i]]] = i;
      ++offset[keys_data[i]];
    }

    result = std::make_pair(count, ret);
  });
  return result;
}

ReorderNodesResult ReorderNodes(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &node_keys_by_ntype,
    const std::vector<int> &num_keys_by_ntype) {
  const int64_t num_ntypes = graph->NumVertexTypes();
  const int64_t num_etypes = graph->NumEdgeTypes();

  ReorderNodesResult ret;
  ret.node_order_by_key.resize(num_ntypes);
  ret.num_nodes_by_key.resize(num_ntypes);
  std::vector<IdArray> inv_order(num_ntypes);

  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const IdArray node_keys = node_keys_by_ntype[ntype];
    const int num_keys = num_keys_by_ntype[ntype];
    std::cout << aten::ToDebugString(node_keys) << ' ' << num_keys << std::endl;
    auto sort_result = CountingSort(node_keys, num_keys);
    ret.num_nodes_by_key[ntype] = sort_result.first;
    ret.node_order_by_key[ntype] = sort_result.second;
    std::cout << aten::ToDebugString(sort_result.first) << std::endl;
    std::cout << aten::ToDebugString(sort_result.second) << std::endl;
    inv_order[ntype] = aten::Scatter(
        aten::Range(0, node_keys->shape[0], node_keys->dtype.bits, node_keys->ctx),
        ret.node_order_by_key[ntype]);
    std::cout << aten::ToDebugString(inv_order[ntype]) << std::endl;
  }

  const auto meta_graph = graph->meta_graph();

  std::vector<HeteroGraphPtr> rel_graphs(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const EdgeArray &edges = graph->Edges(etype, "eid");

    std::cout << aten::ToDebugString(inv_order[srctype]) << std::endl;
    std::cout << aten::ToDebugString(inv_order[dsttype]) << std::endl;
    std::cout << aten::ToDebugString(edges.src) << std::endl;
    std::cout << aten::ToDebugString(edges.dst) << std::endl;
    IdArray new_src = aten::IndexSelect(inv_order[srctype], edges.src);
    IdArray new_dst = aten::IndexSelect(inv_order[dsttype], edges.dst);
    std::cout << aten::ToDebugString(new_src) << std::endl;
    std::cout << aten::ToDebugString(new_dst) << std::endl;
    rel_graphs[etype] = CreateFromCOO(
        srctype == dsttype ? 1 : 2, graph->NumVertices(srctype), graph->NumVertices(dsttype),
        new_src, new_dst);
  }
  std::vector<int64_t> num_nodes_per_type(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type[ntype] = graph->NumVertices(ntype);

  ret.graph = CreateHeteroGraph(meta_graph, rel_graphs, num_nodes_per_type);
  return ret;
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBlock")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &rhs_nodes = ListValueToVector<IdArray>(args[1]);
    const bool include_rhs_in_lhs = args[2];

    HeteroGraphPtr new_graph;
    std::vector<IdArray> lhs_nodes;
    std::vector<IdArray> induced_edges;
    std::tie(new_graph, lhs_nodes, induced_edges) = ToBlock(
        graph_ref.sptr(), rhs_nodes, include_rhs_in_lhs);

    List<Value> lhs_nodes_ref;
    for (IdArray &array : lhs_nodes)
      lhs_nodes_ref.push_back(Value(MakeValue(array)));
    List<Value> induced_edges_ref;
    for (IdArray &array : induced_edges)
      induced_edges_ref.push_back(Value(MakeValue(array)));

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(new_graph));
    ret.push_back(lhs_nodes_ref);
    ret.push_back(induced_edges_ref);

    *rv = ret;
  });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLReorderNodes")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &node_keys_by_ntype = ListValueToVector<IdArray>(args[1]);
    const std::vector<int> &num_keys_by_ntype = ListValueToVector<int>(args[2]);

    auto ret = ReorderNodes(graph_ref.sptr(), node_keys_by_ntype, num_keys_by_ntype);

    List<Value> node_order_by_key_ref;
    for (IdArray &array : ret.node_order_by_key)
      node_order_by_key_ref.push_back(Value(MakeValue(array)));
    List<Value> num_nodes_by_key_ref;
    for (IdArray &array : ret.num_nodes_by_key)
      num_nodes_by_key_ref.push_back(Value(MakeValue(array)));

    List<ObjectRef> result;
    result.push_back(HeteroGraphRef(ret.graph));
    result.push_back(node_order_by_key_ref);
    result.push_back(num_nodes_by_key_ref);

    *rv = result;
  });

};  // namespace transform

};  // namespace dgl
