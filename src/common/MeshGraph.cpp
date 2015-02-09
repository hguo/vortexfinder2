#include "MeshGraph.h"
#include "MeshGraph.pb.h"
#include <cassert>

EdgeIdType2 AlternateEdge(EdgeIdType2 e, int chirality)
{
  if (chirality>0)
    return e;
  else 
    return std::make_tuple(std::get<1>(e), std::get<0>(e));
}

FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality)
{
  using namespace std;

  if (chirality>0) {
    switch (rotation) {
    case 0: return f; 
    case 1: return make_tuple(get<2>(f), get<0>(f), get<1>(f));
    case 2: return make_tuple(get<1>(f), get<2>(f), get<0>(f));
    default: assert(false);
    }
  } else {
    switch (rotation) {
    case 0: return make_tuple(get<2>(f), get<1>(f), get<0>(f));
    case 1: return make_tuple(get<0>(f), get<2>(f), get<1>(f));
    case 2: return make_tuple(get<1>(f), get<0>(f), get<2>(f));
    default: assert(false);
    }
  }
}

FaceIdType4 AlternateFace(FaceIdType4 f, int rotation, int chirality)
{
  using namespace std;

  if (chirality>0) {
    switch (rotation) {
    case 0: return f;
    case 1: return make_tuple(get<3>(f), get<0>(f), get<1>(f), get<2>(f));
    case 2: return make_tuple(get<2>(f), get<3>(f), get<0>(f), get<1>(f));
    case 3: return make_tuple(get<1>(f), get<2>(f), get<3>(f), get<0>(f));
    default: assert(false);
    }
  } else {
    switch (rotation) {
    case 0: return make_tuple(get<3>(f), get<2>(f), get<1>(f), get<0>(f));
    case 1: return make_tuple(get<0>(f), get<3>(f), get<2>(f), get<1>(f));
    case 2: return make_tuple(get<1>(f), get<0>(f), get<3>(f), get<2>(f));
    case 3: return make_tuple(get<2>(f), get<1>(f), get<0>(f), get<3>(f));
    default: assert(false);
    }
  }
}

////////////////////////
MeshGraph::~MeshGraph()
{
  Clear();
}

void MeshGraph::Clear()
{
}

void MeshGraph::SerializeToString(std::string &str) const
{
  PBMeshGraph pmg;

#if 0
  for (int i=0; i<edges.size(); i++) {
    PBEdge *pedge = pmg.add_edges();
    pedge->set_node0( edges[i]->node0 );
    pedge->set_node1( edges[i]->node1 );

    for (int j=0; j<edges[i]->contained_faces.size(); j++) {
      pedge->add_contained_faces( edges[i]->contained_faces[j]->id );
      pedge->add_contained_faces_chirality( edges[i]->contained_faces_chirality[j] );
      pedge->add_contained_faces_eid( edges[i]->contained_faces_eid[j] );
    }
  }

  for (int i=0; i<faces.size(); i++) {
    PBFace *pface = pmg.add_faces();

    for (int j=0; j<faces[i].nodes.size(); j++) 
      pface->add_nodes(faces[i]->nodes[j]);

    for (int j=0; j<faces[i].edges.size(); j++) {
      pface->add_edges(faces[i]->edges[j]->id);
    }
  }
#endif
}

MeshGraphBuilder::MeshGraphBuilder(MeshGraph& mg)
  : _mg(mg)
{
}

EdgeIdType MeshGraphBuilder_Tet::GetEdge(EdgeIdType2 e2, int &chirality)
{
  for (chirality=-1; chirality<2; chirality+=2) {
    std::map<EdgeIdType2, EdgeIdType>::iterator it = _edge_map.find(AlternateEdge(e2, chirality)); 
    if (it != _edge_map.end())
      return it->second;
  }
  return UINT_MAX;
}

FaceIdType MeshGraphBuilder_Tet::GetFace(FaceIdType3 f3, int &chirality)
{
  for (chirality=-1; chirality<2; chirality+=2) 
    for (int rotation=0; rotation<3; rotation++) {
      std::map<FaceIdType3, FaceIdType>::iterator it = _face_map.find(AlternateFace(f3, rotation, chirality));
      if (it != _face_map.end())
        return it->second;
    }
  return UINT_MAX;
}

EdgeIdType MeshGraphBuilder_Tet::AddEdge(EdgeIdType2 e2, int &chirality, FaceIdType f, int eid)
{
  EdgeIdType e = GetEdge(e2, chirality);

  if (e == UINT_MAX) {
    e = _mg.edges.size();
    _edge_map.insert(std::pair<EdgeIdType2, EdgeIdType>(e2, e));
    
    CEdge edge1;
    edge1.node0 = std::get<0>(e2);
    edge1.node1 = std::get<1>(e2);
    
    _mg.edges.push_back(edge1);
    chirality = 1;
  }
  
  CEdge &edge = _mg.edges[e];

  edge.contained_faces.push_back(f);
  edge.contained_faces_chirality.push_back(chirality);
  edge.contained_faces_eid.push_back(eid);
  
  return e;
}

FaceIdType MeshGraphBuilder_Tet::AddFace(FaceIdType3 f3, int &chirality, CellIdType c, int fid)
{
  FaceIdType f = GetFace(f3, chirality);

  if (f == UINT_MAX) {
    f = _face_map.size();
    _face_map.insert(std::pair<FaceIdType3, FaceIdType>(f3, f));
    
    CFace face1;
    face1.nodes.push_back(std::get<0>(f3));
    face1.nodes.push_back(std::get<1>(f3));
    face1.nodes.push_back(std::get<2>(f3));

    EdgeIdType2 e2[3] = {
      std::make_tuple(std::get<0>(f3), std::get<1>(f3)),
      std::make_tuple(std::get<1>(f3), std::get<2>(f3)),
      std::make_tuple(std::get<2>(f3), std::get<0>(f3))};

    for (int i=0; i<3; i++) {
      EdgeIdType e = AddEdge(e2[i], chirality, f, i);
      face1.edges.push_back(e);
      face1.edges_chirality.push_back(chirality);
    }
    
    _mg.faces.push_back(face1);
    chirality = 1;
  }
  
  CFace &face = _mg.faces[f];

  if (chirality == 1) {
    face.contained_cell1 = c;
    face.contained_cell1_fid = fid;
  } else if (chirality == -1) {
    face.contained_cell0 = c;
    face.contained_cell0_fid = fid;
  }

  return f;
}

CellIdType MeshGraphBuilder_Tet::AddCell(
    const std::vector<NodeIdType> &nodes, 
    const std::vector<CellIdType> &neighbors, 
    const std::vector<FaceIdType3> &faces)
{
  CellIdType c = _mg.cells.size();
  CCell cell;

  // nodes
  cell.nodes = nodes;

  // neighbor cells
  cell.neighbor_cells = neighbors;

  // faces and edges
  for (int i=0; i<faces.size(); i++) {
    int chirality; 
    FaceIdType3 f3 = faces[i];

    FaceIdType f = AddFace(f3, chirality, c, i);
    cell.faces.push_back(f);
    cell.faces_chirality.push_back(chirality);
  }

  _mg.cells.push_back(cell);
  return c;
}

#if 0
void MeshGraphBuilder_Tet::Build()
{
  // reorganize to vector
  FaceIdType i = 0;
  for (std::map<FaceIdType3, CFace*>::const_iterator it = _face_map.begin(); 
       it != _face_map.end(); it ++) 
  {
    it->second->id = i ++;
    _mg.faces.push_back(it->second);
  }
  _face_map.clear();

  EdgeIdType j = 0;
  for (std::map<EdgeIdType2, CEdge*>::const_iterator it = _edge_map.begin();
       it != _edge_map.end(); it ++)
  {
    it->second->id = j ++;
    _mg.edges.push_back(it->second);
  }
  _edge_map.clear();
}
#endif
