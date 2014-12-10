#include "gldataset.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

GLDataset::GLDataset()
  : _Jx(0), _Kex(0), _Kex_dot(0), _fluctuation_amp(0), 
    _timestep(0)
{
  memset(_lengths, 0, sizeof(double)*3); 
  memset(_B, 0, sizeof(double)*3); 
}

GLDataset::~GLDataset()
{
}

void GLDataset::SetMagneticField(const double B[3])
{
  memcpy(_B, B, sizeof(double)*3); 
}

void GLDataset::SetKex(double Kex)
{
  _Kex = Kex; 
}

bool GLDataset::OpenDataFile(const std::string& filename)
{
  // no impl
  return false;
}

void GLDataset::LoadTimeStep(int)
{
  // no impl
}

void GLDataset::CloseDataFile()
{
  // no impl
}
