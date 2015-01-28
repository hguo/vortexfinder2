#include "MeshWrapper.h"
#include <cassert>

using namespace libMesh;

EdgeIdType AlternateEdge(EdgeIdType s, int chirality)
{
  if (chirality)
    return s;
  else 
    return std::make_tuple(std::get<1>(s), std::get<0>(s));
}

FaceIdType AlternateFace(FaceIdType f, int rotation, int chirality)
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
  for (std::map<FaceIdType, Face*>::iterator it = _face_map.begin(); 
       it != _face_map.end(); it ++)
  {
    delete it->second;
  }
  _face_map.clear();

  for (std::map<EdgeIdType, Edge*>::iterator it = _side_map.begin();
       it != _side_map.end(); it ++)
  {
    delete it->second;
  }
  _side_map.clear();
}

Edge* MeshWrapper::GetEdge(EdgeIdType s, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) {
    std::map<EdgeIdType, Edge*>::iterator it = _side_map.find(AlternateEdge(s, chirality)); 
    if (it != _side_map.end())
      return it->second;
  }
  return NULL;
}

Edge* MeshWrapper::AddEdge(EdgeIdType s, const Face* f)
{
  int chirality;
  Edge *side = GetEdge(s, chirality);

  if (side == NULL) {
    side = new Edge;
    side->nodes.push_back(std::get<0>(s));
    side->nodes.push_back(std::get<1>(s));
    _side_map.insert(std::pair<EdgeIdType, Edge*>(s, side));
  }

  side->faces.push_back(f);
  return side;
}

Face* MeshWrapper::GetFace(FaceIdType f, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) 
    for (int rotation=0; rotation<3; rotation++) {
      std::map<FaceIdType, Face*>::iterator it = _face_map.find(AlternateFace(f, rotation, chirality));
      if (it != _face_map.end())
        return it->second;
    }
  return NULL;
}

Face* MeshWrapper::AddFace(const Elem* e, int i)
{
  AutoPtr<Elem> f = e->side(i);
  FaceIdType fid(f->node(0), f->node(1), f->node(2));
  
  int chirality;
  Face *face = GetFace(fid, chirality);

  if (face == NULL) {
    face = new Face;
    face->nodes.push_back(f->node(0));
    face->nodes.push_back(f->node(1));
    face->nodes.push_back(f->node(2));
    _face_map.insert(std::pair<FaceIdType, Face*>(fid, face));
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

      EdgeIdType s0(f->nodes[0], f->nodes[1]), 
                 s1(f->nodes[1], f->nodes[2]),
                 s2(f->nodes[2], f->nodes[0]);
      f->sides.push_back( AddEdge(s0, f) );
      f->sides.push_back( AddEdge(s1, f) );
      f->sides.push_back( AddEdge(s2, f) );
    }
  }

  // fprintf(stderr, "n_faces=%lu, n_sides=%lu\n", _face_map.size(), _side_map.size());
}
