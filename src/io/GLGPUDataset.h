#ifndef _GLGPU_DATASET_H
#define _GLGPU_DATASET_H

#include "GLDataset.h"
#include "common/Texel.hpp"

class GLGPUDataset : public GLDataset
{
public: 
  GLGPUDataset(); 
  ~GLGPUDataset();
 
  bool OpenDataFile(const std::string& filename); 
  // void LoadTimeStep(int timestep);
  // void CloseDataFile();
  void OpenNetCDFFile(const std::string& filename);
  void WriteNetCDFFile(const std::string& filename);

  void ComputeSupercurrentField();
  void GetSupercurrentField(const double **sc) const;

  void SerializeDataInfoToString(std::string& buf) const;
  
public:
  // ElemId is encoded by the id of left-bottom corner node in the cell
  void ElemId2Idx(unsigned int id, int *idx) const; 
  unsigned int Idx2ElemId(int *idx) const;
 
  void Idx2Pos(const int idx[3], double pos[3]) const;
  void Pos2Id(const double pos[3], int idx[3]) const;
  void Pos2Grid(const double pos[3], double gpos[3]) const; //!< to grid coordinates

  // counter-cloce wise sides facing outer
  void GetFace(int idx[3], int faceType, int faceIdx[4][3]) const;
  double Flux(int faceType) const;

  double GaugeTransformation(const int idx0[3], const int idx1[3]) const;
  double GaugeTransformation(int x0, int y0, int z0, int x1, int y1, int z1) const;

public:
  std::vector<unsigned int> Neighbors(unsigned int elem_id) const;

public:
  void PrintInfo() const; 

  const int* dims() const {return _dims;}
  const bool* pbc() const {return _pbc;}
  const double* CellLengths() const {return _cell_lengths;}

  double dx() const {return _cell_lengths[0];}
  double dy() const {return _cell_lengths[1];}
  double dz() const {return _cell_lengths[2];}

  const double* amp() const {return _amp;}
  const double* phase() const {return _phase;} 

  double amp(int x, int y, int z) const {return texel3D(_amp, _dims, x, y, z);}
  double phase(int x, int y, int z) const {return texel3D(_phase, _dims, x, y, z);}
  double re(int x, int y, int z) const {return texel3D(_re, _dims, x, y, z);}
  double im(int x, int y, int z) const {return texel3D(_im, _dims, x, y, z);}

private:
  int _dims[3]; 
  bool _pbc[3]; 
  double _cell_lengths[3]; 

  double *_re, *_im, *_amp, *_phase;
  double *_scx, *_scy, *_scz, *_scm; // supercurrent field and its magnitude
}; 

#endif
