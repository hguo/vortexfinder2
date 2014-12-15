#ifndef _GLGPU_DATASET_H
#define _GLGPU_DATASET_H

#include "GLDataset.h"
#include "common/texel.hpp"

enum {
  GLGPU_ENDIAN_LITTLE = 0, 
  GLGPU_ENDIAN_BIG = 1
};

enum {
  GLGPU_TYPE_FLOAT = 0, 
  GLGPU_TYPE_DOUBLE = 1
};

class GLGPUDataset : public GLDataset
{
public: 
  GLGPUDataset(); 
  ~GLGPUDataset();
 
  bool OpenDataFile(const std::string& filename); 
  // void LoadTimeStep(int timestep);
  // void CloseDataFile();
  
  void SerializeDataInfoToString(std::string& buf) const;
  
public:
  // ElemId is encoded by the id of left-bottom corner node in the cell
  void ElemId2Idx(unsigned int id, int *idx) const; 
  unsigned int Idx2ElemId(int *idx) const;
 
  void Idx2Pos(int *idx, double *pos) const;
  void Pos2Id(double *pos, int *idx) const;

  // counter-cloce wise sides facing outer
  void GetFace(int idx[3], int faceType, int faceIdx[4][3]) const;
  double Flux(int faceType) const;

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
}; 

#endif
