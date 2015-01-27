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

class MeshWrapper : public libMesh::Mesh
{
public:
  explicit MeshWrapper(const libMesh::Parallel::Communicator &comm, unsigned char dim=1) : 
    libMesh::Mesh(comm, dim) {}
  
  ~MeshWrapper() {}
};

#endif
