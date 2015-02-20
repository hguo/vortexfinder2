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

void GLGPUDataset::PrintInfo() const
{
  fprintf(stderr, "dims={%d, %d, %d}\n", _dims[0], _dims[1], _dims[2]); 
  fprintf(stderr, "pbc={%d, %d, %d}\n", _pbc[0], _pbc[1], _pbc[2]); 
  fprintf(stderr, "origins={%f, %f, %f}\n", _origins[0], _origins[1], _origins[2]);
  fprintf(stderr, "lengths={%f, %f, %f}\n", _lengths[0], _lengths[1], _lengths[2]);
  fprintf(stderr, "cell_lengths={%f, %f, %f}\n", _cell_lengths[0], _cell_lengths[1], _cell_lengths[2]); 
  fprintf(stderr, "B={%f, %f, %f}\n", _B[0], _B[1], _B[2]);
  fprintf(stderr, "Kex=%f, Kex_dot=%f\n", _Kex, _Kex_dot); 
  fprintf(stderr, "Jxext=%f\n", _Jxext);
  fprintf(stderr, "V=%f\n", _V);
  // fprintf(stderr, "time=%f\n", time); 
  fprintf(stderr, "fluctuation_amp=%f\n", _fluctuation_amp); 
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

bool GLGPUDataset::A(const double X[3], double A[3], int slot) const
{
  A[0] = Ax(X); 
  A[1] = Ay(X); 
  A[2] = Az(X); 
  return true;
}

bool GLGPUDataset::A(NodeIdType n, double A_[3], int slot) const
{
  double X[3];
  Pos(n, X);
  return A(X, A_, slot);
}
