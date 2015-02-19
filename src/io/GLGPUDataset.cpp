#include "GLGPUDataset.h"
#include "GLGPU_IO_Helper.h"

GLGPUDataset::GLGPUDataset() :
  _re(NULL), _im(NULL), 
  _Jx(NULL), _Jy(NULL), _Jz(NULL)
{
}

GLGPUDataset::~GLGPUDataset()
{
}

bool GLGPUDataset::OpenDataFile(const std::string &filename)
{
  bool succ = false;

  if (OpenBDATDataFile(filename)) succ = true; 
  else if (OpenLegacyDataFile(filename)) succ = true;

  return succ;
}

bool GLGPUDataset::OpenLegacyDataFile(const std::string& filename)
{
  // TODO
  return false;
}

bool GLGPUDataset::OpenBDATDataFile(const std::string& filename)
{
  int ndims;
  if (!::GLGPU_IO_Helper_ReadBDAT(
      filename, ndims, _dims, _lengths, _pbc, _B, 
      _Jxext, _Kex, _V, &_re, &_im)) 
    return false;
  
  for (int i=0; i<ndims; i++) {
    _origins[i] = -0.5*_lengths[i];
    if (_pbc[i]) _cell_lengths[i] = _lengths[i] / _dims[i];  
    else _cell_lengths[i] = _lengths[i] / (_dims[i]-1); 
  }

  return true;
}

