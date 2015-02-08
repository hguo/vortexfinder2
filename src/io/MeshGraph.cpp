#include "MeshGraph.h"
#include <cassert>

EdgeIdType2 AlternateEdge(EdgeIdType2 e, int chirality)
{
  if (chirality)
    return e;
  else 
    return std::make_tuple(std::get<1>(e), std::get<0>(e));
}

FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality)
{
  using namespace std;

  if (chirality) {
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

  if (chirality) {
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
CMeshGraphBuilder::CMeshGraphBuilder(ElemIdType n_elems, CMeshGraph& mg)
  : _mg(mg)
{
  for (ElemIdType i=0; i<n_elems; i++)
    mg.elems.push_back(new CElem);
}

CEdge* CMeshGraphBuilder_Tet::GetEdge(EdgeIdType2 e, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) {
    std::map<EdgeIdType2, CEdge*>::iterator it = _edge_map.find(AlternateEdge(e, chirality)); 
    if (it != _edge_map.end())
      return it->second;
  }
  return NULL;
}

CFace* CMeshGraphBuilder_Tet::GetFace(FaceIdType3 f, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) 
    for (int rotation=0; rotation<3; rotation++) {
      std::map<FaceIdType3, CFace*>::iterator it = _face_map.find(AlternateFace(f, rotation, chirality));
      if (it != _face_map.end())
        return it->second;
    }
  return NULL;
}

CEdge* CMeshGraphBuilder_Tet::AddEdge(EdgeIdType2 e, int &chirality, const CFace *f, int eid)
{
  CEdge *edge = GetEdge(e, chirality);

  if (edge == NULL) {
    edge = new CEdge;
    edge->node0 = std::get<0>(e);
    edge->node1 = std::get<1>(e);
    _edge_map.insert(std::pair<EdgeIdType2, CEdge*>(e, edge));
    chirality = 1;
  }

  edge->neighbor_faces.push_back(f);
  edge->neighbor_face_chiralities.push_back(chirality);
  edge->neighbor_face_eid.push_back(eid);
  
  return edge;
}

CFace* CMeshGraphBuilder_Tet::AddFace(FaceIdType3 f, int &chirality, const CElem *el, int fid)
{
  CFace *face = GetFace(fid, chirality);

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
    face->neighbor_elem1 = el;
    face->neighbor_elem1_fid = fid;
  } else if (chirality == -1) {
    face->neighbor_elem0 = el;
    face->neighbor_elem0_fid = fid;
  }

  return face;
}

CElem* CMeshGraphBuilder_Tet::AddElem(ElemIdType i,
    const std::vector<NodeIdType> &nodes, 
    const std::vector<ElemIdType> &neighbors, 
    const std::vector<FaceIdType3> &faces)
{
  CElem *elem = _mg.elems[i];
  _mg.elems[i] = elem;

  // nodes
  elem->nodes = nodes;

  // neighbor elems
  for (int i=0; i<neighbors.size(); i++) {
    if (neighbors[i] != UINT_MAX)
      elem->neighbor_elems.push_back(_mg.elems[neighbors[i]]);
    else 
      elem->neighbor_elems.push_back(NULL);
  }

  // faces and edges
  for (int i=0; i<faces.size(); i++) {
    int chirality; 
    FaceIdType3 fid = faces[i];

    CFace *face = AddFace(fid, chirality, elem, i);
    elem->faces.push_back(face);
    elem->face_chiralities.push_back(chirality);
  }

  return elem;
}

void CMeshGraphBuilder_Tet::Build()
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
