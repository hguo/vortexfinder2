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

typedef std::tuple<NodeIdType, NodeIdType> EdgeIdType2;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType> FaceIdType3;

EdgeIdType2 AlternateEdge(EdgeIdType2 s, int chirality);
FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality);

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

  EdgeIdType NrEdges() const;
  FaceIdType NrFaces() const;

  const Edge* GetEdge(EdgeIdType i) const;
  const Face* GetFace(FaceIdType i) const;

protected: // only used in initialization stage
  Edge* GetEdge(EdgeIdType2 s, int &chirality);
  Face* GetFace(FaceIdType3 f, int &chirality);

  Edge* AddEdge(EdgeIdType2 s, const Face* f);
  Face* AddFace(const libMesh::Elem* elem, int faceId);

protected:
  std::map<EdgeIdType2, Edge*> _edge_map;
  std::map<FaceIdType3, Face*> _face_map;

protected:
  std::vector<Edge*> _edges;
  std::vector<Face*> _faces;
};

/////
struct Edge {
  EdgeIdType id;
  std::vector<NodeIdType> nodes;
  std::vector<const Face*> faces; // faces which contains this edge
};

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
