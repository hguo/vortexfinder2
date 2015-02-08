#include "MeshGraph.h"
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
  for (int i=0; i<edges.size(); i++) 
    delete edges[i]; 

  for (int i=0; i<faces.size(); i++)
    delete faces[i];

  for (int i=0; i<cells.size(); i++)
    delete cells[i];
}
  
MeshGraphBuilder::MeshGraphBuilder(CellIdType n_cells, MeshGraph& mg)
  : _mg(mg)
{
  for (CellIdType i=0; i<n_cells; i++)
    mg.cells.push_back(new CCell);
}

CEdge* MeshGraphBuilder_Tet::GetEdge(EdgeIdType2 e, int &chirality)
{
  for (chirality=-1; chirality<2; chirality+=2) {
    std::map<EdgeIdType2, CEdge*>::iterator it = _edge_map.find(AlternateEdge(e, chirality)); 
    if (it != _edge_map.end())
      return it->second;
  }
  return NULL;
}

CFace* MeshGraphBuilder_Tet::GetFace(FaceIdType3 f, int &chirality)
{
  for (chirality=-1; chirality<2; chirality+=2) 
    for (int rotation=0; rotation<3; rotation++) {
      std::map<FaceIdType3, CFace*>::iterator it = _face_map.find(AlternateFace(f, rotation, chirality));
      if (it != _face_map.end())
        return it->second;
    }
  return NULL;
}

CEdge* MeshGraphBuilder_Tet::AddEdge(EdgeIdType2 e, int &chirality, const CFace *f, int eid)
{
  CEdge *edge = GetEdge(e, chirality);

  if (edge == NULL) {
    edge = new CEdge;
    edge->node0 = std::get<0>(e);
    edge->node1 = std::get<1>(e);
    _edge_map.insert(std::pair<EdgeIdType2, CEdge*>(e, edge));
    chirality = 1;
  }

  edge->contained_faces.push_back(f);
  edge->contained_face_chiralities.push_back(chirality);
  edge->contained_face_eid.push_back(eid);
  
  return edge;
}

CFace* MeshGraphBuilder_Tet::AddFace(FaceIdType3 f, int &chirality, const CCell *el, int fid)
{
  CFace *face = GetFace(f, chirality);

  if (face == NULL) {
    face = new CFace;
    face->nodes.push_back(std::get<0>(f));
    face->nodes.push_back(std::get<1>(f));
    face->nodes.push_back(std::get<2>(f));
    _face_map.insert(std::pair<FaceIdType3, CFace*>(f, face));

    EdgeIdType2 e[3] = {
      std::make_tuple(std::get<0>(f), std::get<1>(f)),
      std::make_tuple(std::get<1>(f), std::get<2>(f)),
      std::make_tuple(std::get<2>(f), std::get<0>(f))};

    for (int i=0; i<3; i++) {
      CEdge *edge = AddEdge(e[i], chirality, face, i);
      face->edges.push_back(edge);
      face->edge_chiralities.push_back(chirality);
    }

    chirality = 1;
  }

  if (chirality == 1) {
    face->contained_cell1 = el;
    face->contained_cell1_fid = fid;
  } else if (chirality == -1) {
    face->contained_cell0 = el;
    face->contained_cell0_fid = fid;
  }

  return face;
}

CCell* MeshGraphBuilder_Tet::AddCell(CellIdType i,
    const std::vector<NodeIdType> &nodes, 
    const std::vector<CellIdType> &neighbors, 
    const std::vector<FaceIdType3> &faces)
{
  CCell *cell = _mg.cells[i];
  _mg.cells[i] = cell;

  // nodes
  cell->nodes = nodes;

  // neighbor cells
  for (int i=0; i<neighbors.size(); i++) {
    if (neighbors[i] != UINT_MAX)
      cell->neighbor_cells.push_back(_mg.cells[neighbors[i]]);
    else 
      cell->neighbor_cells.push_back(NULL);
  }

  // faces and edges
  for (int i=0; i<faces.size(); i++) {
    int chirality; 
    FaceIdType3 fid = faces[i];

    CFace *face = AddFace(fid, chirality, cell, i);
    cell->faces.push_back(face);
    cell->face_chiralities.push_back(chirality);
  }

  return cell;
}

void MeshGraphBuilder_Tet::Build()
{
  // reorganize to vector
  FaceIdType i = 0;
  for (std::map<FaceIdType3, CFace*>::const_iterator it = _face_map.begin(); 
       it != _face_map.end(); it ++) 
  {
    // it->second->id = i ++;
    _mg.faces.push_back(it->second);
  }
  _face_map.clear();

  EdgeIdType j = 0;
  for (std::map<EdgeIdType2, CEdge*>::const_iterator it = _edge_map.begin();
       it != _edge_map.end(); it ++)
  {
    // it->second->id = j ++;
    _mg.edges.push_back(it->second);
  }
  _edge_map.clear();
}
