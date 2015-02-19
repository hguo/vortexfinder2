#include "GLGPUDataset.h"
#include "GLGPU_IO_Helper.h"
#include <cassert>
#include <glob.h>

GLGPUDataset::GLGPUDataset() :
  _re(NULL), _im(NULL), 
  _re1(NULL), _im1(NULL),
  _Jx(NULL), _Jy(NULL), _Jz(NULL)
{
}

GLGPUDataset::~GLGPUDataset()
{
  if (_re != NULL) delete _re;
  if (_im != NULL) delete _im;
  if (_re1 != NULL) delete _re1;
  if (_im1 != NULL) delete _im1;
}

bool GLGPUDataset::OpenDataFile(const std::string &pattern)
{
  glob_t results;

  glob(pattern.c_str(), 0, NULL, &results);
  _filenames.clear();
  for (int i=0; i<results.gl_pathc; i++) 
    _filenames.push_back(results.gl_pathv[i]);

  return _filenames.size()>0;
}

void GLGPUDataset::CloseDataFile()
{
  _filenames.clear();
}

void GLGPUDataset::LoadTimeStep(int timestep, int slot)
{
  assert(timestep>=0 && timestep<=_filenames.size());
  bool succ = false;
  const std::string &filename = _filenames[timestep];

  // rotate
  if (slot>0) {
    double *r = _re, *i = _im;
    _re = _re1; _im = _im1;
    _re1 = r; _im1 = i;
  }

  // load
  if (OpenBDATDataFile(filename, slot)) succ = true; 
  else if (OpenLegacyDataFile(filename, slot)) succ = true;

  SetTimeStep(timestep, slot);
}

bool GLGPUDataset::OpenLegacyDataFile(const std::string& filename, int time)
{
  // TODO
  return false;
}

bool GLGPUDataset::OpenBDATDataFile(const std::string& filename, int slot)
{
  int ndims;
  if (!::GLGPU_IO_Helper_ReadBDAT(
      filename, ndims, _dims, _lengths, _pbc, _B, 
      _Jxext, _Kex, _V, 
      slot == 0? &_re : &_re1, 
      slot == 0? &_im : &_im1)) 
    return false;
  
  for (int i=0; i<ndims; i++) {
    _origins[i] = -0.5*_lengths[i];
    if (_pbc[i]) _cell_lengths[i] = _lengths[i] / _dims[i];  
    else _cell_lengths[i] = _lengths[i] / (_dims[i]-1); 
  }

  return true;
}

