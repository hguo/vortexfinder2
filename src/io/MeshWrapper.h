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

typedef std::tuple<NodeIdType, NodeIdType> SideIdType;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType> FaceIdType;

SideIdType AlternateSide(SideIdType s, int chirality);
FaceIdType AlternateFace(FaceIdType f, int rotation, int chirality);

struct Side;
struct Face;

class MeshWrapper : public libMesh::Mesh
{
public:
  explicit MeshWrapper(const libMesh::Parallel::Communicator &comm, unsigned char dim=1) : 
    libMesh::Mesh(comm, dim) {}
  
  ~MeshWrapper() {}

public:
  Side* GetSide(SideIdType s, int &chirality);
  Side* AddSide(SideIdType s, const Face* f);

  Face* GetFace(FaceIdType f, int &chirality);
  Face* AddFace(FaceIdType f, const libMesh::Elem* elem);

protected:
  std::map<FaceIdType, Face*> _face_map;
  std::map<SideIdType, Side*> _side_map;
};

/////
struct Side {
  std::vector<const Face*> faces;
};

struct Face {
  // std::vector<const Elem*> elems; // a face could only be shared by two elements
  const libMesh::Elem *elem_front, *elem_back; // front: same chirality; back: opposite chirality
  std::vector<const Side*> sides;

  explicit Face() : elem_front(NULL), elem_back(NULL) {}
};

#endif
