#ifndef _CONDOR2_DATASET_H
#define _CONDOR2_DATASET_H

#include "GLDataset.h"
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>

class Condor2Dataset : public libMesh::ParallelObject, public GLDataset
{
public:
  Condor2Dataset(const libMesh::Parallel::Communicator &comm); 
  ~Condor2Dataset(); 

public: 
  bool OpenDataFile(const std::string& filename); 
  void LoadTimeStep(int timestep);
  void CloseDataFile();

  // APIs for in-situ analysis
  void SetMesh(libMesh::Mesh*);
  void SetEquationSystems(libMesh::EquationSystems*);

  void PrintInfo() const; 

public:
  libMesh::UnstructuredMesh* mesh() const {return _mesh;}
  libMesh::EquationSystems* eqsys() const {return _eqsys;}
  libMesh::NonlinearImplicitSystem* tsys() const {return _tsys;}

  unsigned int u_var() const {return _u_var;}
  unsigned int v_var() const {return _v_var;}

private:
  libMesh::UnstructuredMesh *_mesh;
  libMesh::ExodusII_IO *_exio; 
  libMesh::EquationSystems *_eqsys;
  libMesh::NonlinearImplicitSystem *_tsys;
  
  unsigned int _u_var, _v_var;
}; 

#endif
