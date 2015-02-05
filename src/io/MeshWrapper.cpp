#include "MeshWrapper.h"
#include <cassert>

using namespace libMesh;

EdgeIdType2 AlternateEdge(EdgeIdType2 s, int chirality)
{
  if (chirality)
    return s;
  else 
    return std::make_tuple(std::get<1>(s), std::get<0>(s));
}

FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality)
{
  using namespace std;

  if (chirality) {
    switch (rotation) {
    case 0: return make_tuple(get<0>(f), get<1>(f), get<2>(f)); // nochange
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

///////
MeshWrapper::~MeshWrapper()
{
  for (FaceIdType i=0; i<_faces.size(); i++)
    delete _faces[i];
  _faces.clear();

  for (EdgeIdType i=0; i<_edges.size(); i++)
    delete _edges[i];
  _edges.clear();
}

Edge* MeshWrapper::GetEdge(EdgeIdType2 s, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) {
    std::map<EdgeIdType2, Edge*>::iterator it = _edge_map.find(AlternateEdge(s, chirality)); 
    if (it != _edge_map.end())
      return it->second;
  }
  return NULL;
}

Edge* MeshWrapper::AddEdge(EdgeIdType2 s, const Face* f)
{
  int chirality;
  Edge *edge = GetEdge(s, chirality);

  if (edge == NULL) {
    edge = new Edge;
    edge->nodes.push_back(std::get<0>(s));
    edge->nodes.push_back(std::get<1>(s));
    _edge_map.insert(std::pair<EdgeIdType2, Edge*>(s, edge));
  }

  edge->faces.push_back(f);
  return edge;
}

Face* MeshWrapper::GetFace(FaceIdType3 f, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) 
    for (int rotation=0; rotation<3; rotation++) {
      std::map<FaceIdType3, Face*>::iterator it = _face_map.find(AlternateFace(f, rotation, chirality));
      if (it != _face_map.end())
        return it->second;
    }
  return NULL;
}

Face* MeshWrapper::AddFace(const Elem* e, int i)
{
  AutoPtr<Elem> f = e->side(i);
  FaceIdType3 fid(f->node(0), f->node(1), f->node(2));
  
  int chirality;
  Face *face = GetFace(fid, chirality);

  if (face == NULL) {
    face = new Face;
    face->nodes.push_back(f->node(0));
    face->nodes.push_back(f->node(1));
    face->nodes.push_back(f->node(2));
    _face_map.insert(std::pair<FaceIdType3, Face*>(fid, face));
  }

  if (chirality) {
    face->elem_front = e->id();
    face->elem_face_front = i;
  } else {
    face->elem_back = e->id();
    face->elem_face_back = i;
  }

  return face;
}

void MeshWrapper::InitializeWrapper()
{
  MeshBase::const_element_iterator it = local_elements_begin(); 
  const MeshBase::const_element_iterator end = local_elements_end(); 

  for (; it!=end; it++) {
    const Elem *e = *it;
    for (int i=0; i<e->n_sides(); i++) {
      Face *f = AddFace(e, i);

      EdgeIdType2 s0(f->nodes[0], f->nodes[1]), 
                  s1(f->nodes[1], f->nodes[2]),
                  s2(f->nodes[2], f->nodes[0]);
      f->edges.push_back( AddEdge(s0, f) );
      f->edges.push_back( AddEdge(s1, f) );
      f->edges.push_back( AddEdge(s2, f) );
    }
  }

  // reorganize to vector
  FaceIdType i = 0;
  for (std::map<FaceIdType3, Face*>::const_iterator it = _face_map.begin(); 
       it != _face_map.end(); it ++) 
  {
    it->second->id = i ++;
    _faces.push_back(it->second);
  }
  _face_map.clear();

  EdgeIdType j = 0;
  for (std::map<EdgeIdType2, Edge*>::const_iterator it = _edge_map.begin();
       it != _edge_map.end(); it ++)
  {
    it->second->id = j ++;
    _edges.push_back(it->second);
  }
  _edge_map.clear();
}

const Edge* MeshWrapper::GetEdge(EdgeIdType i) const
{
  return _edges[i];
}

const Face* MeshWrapper::GetFace(FaceIdType i) const
{
  return _faces[i];
}

EdgeIdType MeshWrapper::NrEdges() const
{
  return _edges.size();
}

FaceIdType MeshWrapper::NrFaces() const
{
  return _faces.size();
}
