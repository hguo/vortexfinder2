#ifndef _CONDOR2_DATASET_H
#define _CONDOR2_DATASET_H

#include "GLDataset.h"
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/fe_base.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/exodusII_io.h>

class Condor2Dataset : public libMesh::ParallelObject, public GLDataset
{
public:
  Condor2Dataset(const libMesh::Parallel::Communicator &comm); 
  ~Condor2Dataset(); 

  int Dimensions() const {return 3;}
  int NrFacesPerElem() const {return 4;}
  int NrNodesPerFace() const {return 3;}

public: 
  bool OpenDataFile(const std::string& filename); 
  void LoadTimeStep(int timestep);
  void CloseDataFile();

  // APIs for in-situ analysis
  void SetMesh(libMesh::Mesh*);
  void SetEquationSystems(libMesh::EquationSystems*);

  void PrintInfo() const; 
  void SerializeDataInfoToString(std::string& buf) const;

public:
  std::vector<ElemIdType> GetNeighborIds(ElemIdType id) const;
  bool GetFace(ElemIdType id, int face, double X[][3], double A[][3], double re[], double im[]) const;
  bool OnBoundary(ElemIdType id) const;

public:
  libMesh::UnstructuredMesh* mesh() const {return _mesh;}
  libMesh::EquationSystems* eqsys() const {return _eqsys;}
  libMesh::NonlinearImplicitSystem* tsys() const {return _tsys;}
  libMesh::System* asys() const {return _asys;}

  unsigned int u_var() const {return _u_var;}
  unsigned int v_var() const {return _v_var;}
  unsigned int Ax_var() const {return _Ax_var;}
  unsigned int Ay_var() const {return _Ay_var;}
  unsigned int Az_var() const {return _Az_var;}

public:
  ElemIdType Pos2ElemId(const double X[]) const;
  
  // Order parameters (direct access/linear interpolation)
  bool Psi(const double X[3], double &re, double &im) const;

  // magnetic potential
  bool A(const double X[3], double A[3]) const;

  // Supercurrent field
  bool Supercurrent(const double X[3], double J[3]) const;

private: 
  void ProbeBoundingBox();

private:
  libMesh::UnstructuredMesh *_mesh;
  libMesh::ExodusII_IO *_exio; 
  libMesh::EquationSystems *_eqsys;
  libMesh::NonlinearImplicitSystem *_tsys;
  libMesh::System *_asys;
  
  unsigned int _u_var, _v_var;
  unsigned int _Ax_var, _Ay_var, _Az_var;
  unsigned int _rho_var, _phi_var;
}; 

#endif
