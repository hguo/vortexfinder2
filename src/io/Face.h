#ifndef _FACE_H
#define _FACE_H

#include "def.h"
#include <vector>

struct Edge;

struct Face {
  FaceIdType id;

  // std::vector<ElemIdType> elems; // elements which contains this face
  // const libMesh::Elem *elem_front, *elem_back; // front: same chirality; back: opposite chirality
  ElemIdType elem_front, elem_back;
  int elem_face_front, elem_face_back; // the face id in elements
  std::vector<const Edge*> edges;
  std::vector<NodeIdType> nodes;

  explicit Face() : elem_front(UINT_MAX), elem_back(UINT_MAX) {}
};

#endif
