#ifndef _CONDOR2_DATASET_H
#define _CONDOR2_DATASET_H

#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/elem.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/equation_systems.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/point_locator_tree.h>
#include <libmesh/exodusII_io.h>
#include "GLDataset.h"

class Condor2Dataset : public libMesh::ParallelObject, public GLDataset
{
public:
  Condor2Dataset(const libMesh::Parallel::Communicator &comm); 
  ~Condor2Dataset(); 

  int Dimensions() const {return 3;}

public: 
  bool OpenDataFile(const std::string& filename);
  bool LoadTimeStep(int timestep, int slot);
  void CloseDataFile();

  void BuildMeshGraph();

  // APIs for in-situ analysis
  void SetMesh(libMesh::Mesh*);
  void SetEquationSystems(libMesh::EquationSystems*);

  void PrintInfo(int slot=0) const; 
  void SerializeDataInfoToString(std::string& buf) const;

public:
  void GetFaceValues(const CFace&, int slot, float X[][3], float A[][3], float rho[], float phi[]) const;
  void GetSpaceTimeEdgeValues(const CEdge&, float X[][3], float A[][3], float rho[], float phi[]) const;
  
  CellIdType Pos2CellId(const float X[]) const; //!< returns the elemId for a given position
  bool OnBoundary(ElemIdType id) const;

public:
  libMesh::UnstructuredMesh* mesh() const {return _mesh;}
  libMesh::EquationSystems* eqsys() const {return _eqsys;}
  libMesh::NonlinearImplicitSystem* tsys() const {return _tsys;}
  libMesh::System* asys() const {return _asys;}

  unsigned int rho_var() const {return _rho_var;}
  unsigned int phi_var() const {return _phi_var;}
  unsigned int Ax_var() const {return _Ax_var;}
  unsigned int Ay_var() const {return _Ay_var;}
  unsigned int Az_var() const {return _Az_var;}

public:
  float Rho(NodeIdType, int slot) const;
  float Phi(NodeIdType, int slot) const;

  bool Pos(NodeIdType, float X[3]) const;
  bool Psi(const float X[3], float &re, float &im, int slot) const;
  // bool Psi(NodeIdType, float &re, float &im, int slot) const;
  bool A(const float X[3], float A[3], int slot) const;
  bool A(NodeIdType, float A[3], int slot) const;
  bool Supercurrent(const float X[3], float J[3], int slot) const;
  bool Supercurrent(NodeIdType, float J[3], int slot) const;

private: 
  void ProbeBoundingBox();
  void LoadTimeStep_(int timestep);

private:
  const libMesh::Elem* LocateElemCoherently(const float X[3]) const; // not thread-safe

private:
  libMesh::UnstructuredMesh *_mesh;
  libMesh::ExodusII_IO *_exio; 
  libMesh::EquationSystems *_eqsys;
  libMesh::NonlinearImplicitSystem *_tsys;
  libMesh::System *_asys;
  libMesh::PointLocatorTree *_locator;

  unsigned int _rho_var, _phi_var;
  unsigned int _Ax_var, _Ay_var, _Az_var;
  // unsigned int _rho_var, _phi_var;

  libMesh::AutoPtr<libMesh::NumericVector<libMesh::Number> > _ts, _ts1;
  libMesh::AutoPtr<libMesh::NumericVector<libMesh::Number> > _as, _as1;
}; 

#endif
