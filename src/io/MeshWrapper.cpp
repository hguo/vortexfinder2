#include "MeshWrapper.h"
#include <cassert>

using namespace libMesh;

SideIdType AlternateSide(SideIdType s, int chirality)
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
Side* MeshWrapper::GetSide(SideIdType s, int &chirality)
{
  for (chirality=0; chirality<2; chirality++) {
    std::map<SideIdType, Side*>::iterator it = _side_map.find(AlternateSide(s, chirality)); 
    if (it != _side_map.end())
      return it->second;
  }
  return NULL;
}

Side* MeshWrapper::AddSide(SideIdType s, const Face* f)
{
  int chirality;
  Side *side = GetSide(s, chirality);

  if (side == NULL) {
    side = new Side;
    _side_map.insert(std::pair<SideIdType, Side*>(s, side));
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

Face* MeshWrapper::AddFace(FaceIdType f, const Elem* elem)
{
  int chirality;
  Face *face = GetFace(f, chirality);

  if (face == NULL) {
    face = new Face;
    _face_map.insert(std::pair<FaceIdType, Face*>(f, face));
  }

  if (chirality)
    face->elem_front = elem;
  else 
    face->elem_back = elem;

  return face;
}
