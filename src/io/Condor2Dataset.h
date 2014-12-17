#ifndef _CONDOR2_DATASET_H
#define _CONDOR2_DATASET_H

#include "GLDataset.h"
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/fe_base.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/point_locator_tree.h>
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

  void ComputeSupercurrentField(); 

  // APIs for in-situ analysis
  void SetMesh(libMesh::Mesh*);
  void SetEquationSystems(libMesh::EquationSystems*);

  void PrintInfo() const; 
  void SerializeDataInfoToString(std::string& buf) const;

public:
  std::vector<unsigned int> Neighbors(unsigned int elem_id) const;

public:
  libMesh::UnstructuredMesh* mesh() const {return _mesh;}
  libMesh::EquationSystems* eqsys() const {return _eqsys;}
  libMesh::NonlinearImplicitSystem* tsys() const {return _tsys;}

  unsigned int u_var() const {return _u_var;}
  unsigned int v_var() const {return _v_var;}

public:
  unsigned int Pos2ElemId(const double X[]) const;
  
  // Order parameters (direct access/linear interpolation)
  bool Psi(const double X[3], double &re, double &im) const;

  // Supercurrent field
  bool Supercurrent(const double X[3], double J[3]) const;

private: 
  void ProbeBoundingBox();

private:
  libMesh::UnstructuredMesh *_mesh;
  libMesh::ExodusII_IO *_exio; 
  libMesh::EquationSystems *_eqsys;
  libMesh::NonlinearImplicitSystem *_tsys;
  libMesh::PointLocatorTree *_locator;
  libMesh::AutoPtr<libMesh::FEBase> _fe;
  
  unsigned int _u_var, _v_var;
}; 

#endif
