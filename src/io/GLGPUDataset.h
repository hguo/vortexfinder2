#ifndef _GLGPUDATASET_H
#define _GLGPUDATASET_H

#include "io/GLDataset.h"

class GLGPUDataset : public GLDataset
{
public:
  GLGPUDataset();
  ~GLGPUDataset();
  
  void SerializeDataInfoToString(std::string& buf) const;
  
public:
  bool OpenDataFile(const std::string& filename); // file list
  bool OpenDataFileByPattern(const std::string& pattern); 
  void LoadTimeStep(int timestep, int slot=0);
  void RotateTimeSteps();
  void CloseDataFile();

  int NTimeSteps() const {return _filenames.size();}

  void PrintInfo(int slot=0) const;
 
  bool BuildDataFromArray(const GLHeader&, const double *psi);

private:
  bool OpenBDATDataFile(const std::string& filename, int slot=0);
  bool OpenLegacyDataFile(const std::string& filename, int slot=0);
  
  void ModulateKex(int slot=0);

protected:
  void Nid2Idx(NodeIdType id, int *idx) const; 
  NodeIdType Idx2Nid(const int *idx) const;
 
  void Idx2Pos(const int idx[3], double X[3]) const;
  void Pos2Idx(const double X[3], int idx[3]) const;
  void Pos2Grid(const double pos[3], double gpos[3]) const; //!< to grid coordinates

public: // rectilinear grid
  const int* dims() const {return _h[0].dims;}
  const bool* pbc() const {return _h[0].pbc;}
  const double* CellLengths() const {return _h[0].cell_lengths;}

  double dx() const {return _h[0].cell_lengths[0];}
  double dy() const {return _h[0].cell_lengths[1];}
  double dz() const {return _h[0].cell_lengths[2];}
  
  // Magnetic potential
  bool A(const double X[3], double A[3], int slot=0) const;
  bool A(NodeIdType n, double A[3], int slot=0) const;
  
  // Magnetic field
  const double* B(int slot=0) const {return _h[slot].B;}

  // bool Psi(NodeIdType, double &rho, double &phi, int slot=0) const;
  // bool Psi(const double X[3], double &rho, double &phi, int slot=0) const;

  inline double Rho(NodeIdType i, int slot=0) const {return _psi[slot][i*2];}
  inline double Phi(NodeIdType i, int slot=0) const {return _psi[slot][i*2+1];}
  
  bool Pos(NodeIdType, double X[3]) const;
  bool Supercurrent(const double X[3], double J[3], int slot=0) const;
  bool Supercurrent(NodeIdType, double J[3], int slot=0) const;

public:
  double QP(const double X0[], const double X1[], int slot=0) const;

protected:
  double *_psi[2]; // (rho, phi)
  double *_Jx, *_Jy, *_Jz; // only for timestep 0

  std::vector<std::string> _filenames; // filenames for different timesteps
};

#endif
