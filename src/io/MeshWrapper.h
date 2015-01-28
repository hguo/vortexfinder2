#ifndef _MESH_WRAPPER_H
#define _MESH_WRAPPER_H

#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/point_locator_tree.h>
#include <libmesh/exodusII_io.h>
#include "def.h"

typedef std::tuple<NodeIdType, NodeIdType> EdgeIdType;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType> FaceIdType;

EdgeIdType AlternateEdge(EdgeIdType s, int chirality);
FaceIdType AlternateFace(FaceIdType f, int rotation, int chirality);

struct Edge;
struct Face;

class MeshWrapper : public libMesh::Mesh
{
public:
  explicit MeshWrapper(const libMesh::Parallel::Communicator &comm, unsigned char dim=1) : 
    libMesh::Mesh(comm, dim) {}
  
  ~MeshWrapper();

public:
  void InitializeWrapper();

  Edge* GetEdge(EdgeIdType s, int &chirality);
  Face* GetFace(FaceIdType f, int &chirality);

public:
  typedef std::map<FaceIdType, Face*>::const_iterator const_face_iterator;
  typedef std::map<FaceIdType, Face*>::iterator face_iterator;

  typedef std::map<EdgeIdType, Edge*>::const_iterator const_side_iterator;
  typedef std::map<EdgeIdType, Edge*>::iterator side_iterator;

  const_face_iterator face_begin() const {return _face_map.begin();}
  const_face_iterator face_end() const {return _face_map.end();}
  face_iterator face_begin() {return _face_map.begin();}
  face_iterator face_end() {return _face_map.end();}

  const_side_iterator side_begin() const {return _side_map.begin();}
  const_side_iterator side_end() const {return _side_map.end();}
  side_iterator side_begin() {return _side_map.begin();}
  side_iterator side_end() {return _side_map.end();}

protected:
  Edge* AddEdge(EdgeIdType s, const Face* f);
  Face* AddFace(const libMesh::Elem* elem, int faceId);

protected:
  std::map<FaceIdType, Face*> _face_map;
  std::map<EdgeIdType, Edge*> _side_map;
};

/////
struct Edge {
  std::vector<NodeIdType> nodes;
  std::vector<const Face*> faces; // faces which contains this side
};

struct Face {
  // std::vector<ElemIdType> elems; // elements which contains this face
  // const libMesh::Elem *elem_front, *elem_back; // front: same chirality; back: opposite chirality
  ElemIdType elem_front, elem_back;
  int elem_face_front, elem_face_back; // the face id in elements
  std::vector<const Edge*> sides;
  std::vector<NodeIdType> nodes;

  explicit Face() : elem_front(UINT_MAX), elem_back(UINT_MAX) {}
};

#endif
