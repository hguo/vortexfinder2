#ifndef _GLGPU_DATASET_H
#define _GLGPU_DATASET_H

#include "GLDataset.h"

enum {
  GLGPU_ENDIAN_LITTLE = 0, 
  GLGPU_ENDIAN_BIG = 1
};

enum {
  GLGPU_TYPE_FLOAT = 0, 
  GLGPU_TYPE_DOUBLE = 1
};

template <typename T>
static T& texel3D(T* p, const int* sizes, int x, int y, int z)
{
  return p[x + sizes[0]*(y + sizes[1]*z)]; 
}


class GLGPUDataset : public GLDataset
{
public: 
  GLGPUDataset(); 
  ~GLGPUDataset();

  bool LoadFromFile(const std::string& filename); //!< "legacy" format
  void WriteToNetCDF(const std::string& filename); 

  const int* dims() const {return _dims;}
  const bool* pbc() const {return _pbc;}
  const double* cellLengths() const {return _cellLengths;}

  const double* amp() const {return _amp;}
  const double* phase() const {return _phase;} 

  double amp(int x, int y, int z) const {return texel3D(_amp, _dims, x, y, z);}
  double phase(int x, int y, int z) const {return texel3D(_phase, _dims, x, y, z);}
  double re(int x, int y, int z) const {return texel3D(_re, _dims, x, y, z);}
  double im(int x, int y, int z) const {return texel3D(_im, _dims, x, y, z);}

private:
  int _dims[3]; 
  bool _pbc[3]; 
  double _cellLengths[3]; 

  double *_re, *_im, *_amp, *_phase; 
}; 

#endif
