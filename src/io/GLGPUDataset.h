#ifndef _GLGPU_DATASET_H
#define _GLGPU_DATASET_H

#include <cmath>
#include "GLDataset.h"
#include "common/Texel.hpp"

class GLGPUDataset : public GLDataset
{
public: 
  GLGPUDataset(); 
  ~GLGPUDataset();
  
  void PrintInfo() const; 
  void SerializeDataInfoToString(std::string& buf) const;

public: // data I/O
  bool OpenDataFile(const std::string& filename); 
  bool OpenLegacyDataFile(const std::string& filename);
  bool OpenBDATDataFile(const std::string& filename);
  bool OpenNetCDFFile(const std::string& filename);
  bool WriteNetCDFFile(const std::string& filename);

public: // mesh info
  int Dimensions() const {return 3;}
  int NrFacesPerElem() const {return 6;}
  int NrNodesPerFace() const {return 4;}
  
public: // mesh utils
  std::vector<ElemIdType> GetNeighborIds(ElemIdType elem_id) const;
  bool GetFace(ElemIdType id, int face, double X[][3], double A[][3], double re[], double im[]) const;
  ElemIdType Pos2ElemId(const double X[]) const; 
  bool OnBoundary(ElemIdType id) const;

  // ElemId is encoded by the id of left-bottom corner node in the cell
  void ElemId2Idx(ElemIdType id, int *idx) const; 
  ElemIdType Idx2ElemId(int *idx) const;
 
  void Idx2Pos(const int idx[3], double pos[3]) const;
  void Pos2Idx(const double pos[3], int idx[3]) const;
  void Pos2Grid(const double pos[3], double gpos[3]) const; //!< to grid coordinates

public: // transformations and utils
  double GaugeTransformation(const double X0[], const double X1[]) const;

  double Flux(int faceType) const;
  double QP(const double X0[], const double X1[]) const;

public: // rectilinear grid
  const int* dims() const {return _dims;}
  const bool* pbc() const {return _pbc;}
  const double* CellLengths() const {return _cell_lengths;}

  double dx() const {return _cell_lengths[0];}
  double dy() const {return _cell_lengths[1];}
  double dz() const {return _cell_lengths[2];}
  
  // Magnetic potential
  bool A(const double X[3], double A[3]) const; //!< compute the vector potential at given position
  double Ax(const double X[3]) const {if (By()>0) return 0; else return -X[1]*Bz();}
  double Ay(const double X[3]) const {if (By()>0) return X[0]*Bz(); else return 0;}
  double Az(const double X[3]) const {if (By()>0) return -X[0]*By(); else return X[1]*Bx();}
  
  // Magnetic field
  void SetMagneticField(const double B[3]);
  const double* B() const {return _B;}
  double Bx() const {return _B[0];} 
  double By() const {return _B[1];} 
  double Bz() const {return _B[2];}

public: // data access
  const double& Re(int x, int y, int z) const {return texel3D(_re, _dims, x, y, z);}
  const double& Im(int x, int y, int z) const {return texel3D(_im, _dims, x, y, z);}
  double Rho(int x, int y, int z) const {double r=Re(x, y, z), i=Im(x, y, z); return sqrt(r*r+i*i);}
  double Phi(int x, int y, int z) const {double r=Re(x, y, z), i=Im(x, y, z); return atan2(i, r);}
  
  // Order parameters (direct access/linear interpolation)
  bool Psi(const double X[3], double &re, double &im) const;

  // Supercurrent field
  bool Supercurrent(const double X[3], double J[3]) const;

protected:
  void Reset();
  void ComputeSupercurrentField();

private:
  int _dims[3]; 
  bool _pbc[3]; 
  double _cell_lengths[3]; 

  double _B[3];

  double *_re, *_im;
  double *_Jx, *_Jy, *_Jz;
}; 

#endif
