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
#include "Edge.h"
#include "Face.h"

typedef std::tuple<NodeIdType, NodeIdType> EdgeIdType2;
typedef std::tuple<NodeIdType, NodeIdType, NodeIdType> FaceIdType3;

EdgeIdType2 AlternateEdge(EdgeIdType2 s, int chirality);
FaceIdType3 AlternateFace(FaceIdType3 f, int rotation, int chirality);

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

  Edge* AddEdge(EdgeIdType2 e, const Face* f, int eid);
  Face* AddFace(const libMesh::Elem* elem, int fid);

protected:
  std::map<EdgeIdType2, Edge*> _edge_map;
  std::map<FaceIdType3, Face*> _face_map;

protected:
  std::vector<Edge*> _edges;
  std::vector<Face*> _faces;
};

#endif
