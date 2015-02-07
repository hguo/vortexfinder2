#ifndef _CONDOR2_DATASET_H
#define _CONDOR2_DATASET_H

#include "GLDataset.h"
#include "MeshWrapper.h"

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
  void LoadNextTimeStep(int span=1);
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

  bool GetFaceValues(const Face* f, double X[][3], double A[][3], double re[], double im[]) const;
  bool GetFacePrismValues(const Face* f, double X[6][3], double A[6][3], double re[6], double im[6]) const;
  
  bool GetSpaceTimeEdgeValues(const Edge*, double X[][3], double A[][3], double re[], double im[]) const;

public:
  // libMesh::UnstructuredMesh* mesh() const {return _mesh;}
  MeshWrapper* mesh() const {return _mesh;}
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
  const libMesh::Elem* LocateElemCoherently(const double X[3]) const; // not thread-safe

private:
  // libMesh::UnstructuredMesh *_mesh;
  MeshWrapper *_mesh;
  libMesh::ExodusII_IO *_exio; 
  libMesh::EquationSystems *_eqsys;
  libMesh::NonlinearImplicitSystem *_tsys;
  libMesh::System *_asys;
  libMesh::PointLocatorTree *_locator;

  unsigned int _u_var, _v_var;
  unsigned int _Ax_var, _Ay_var, _Az_var;
  // unsigned int _rho_var, _phi_var;

  libMesh::AutoPtr<libMesh::NumericVector<libMesh::Number> > _tsolution, _tsolution1;
  libMesh::AutoPtr<libMesh::NumericVector<libMesh::Number> > _asolution, _asolution1;
}; 

#endif
