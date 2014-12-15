#include "gldataset.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

GLDataset::GLDataset()
  : _Kex(0), _Kex_dot(0), _fluctuation_amp(0), 
    _timestep(0)
{
  memset(_origins, 0, sizeof(double)*3); 
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
  
double GLDataset::GaugeTransformation(const double *X0, const double *X1) const
{
  double gx, gy, gz; 
  double dx = X1[0] - X0[0], 
         dy = X1[1] - X0[1], 
         dz = X1[2] - X0[2]; 
  double x = X0[0] + 0.5*dx, 
         y = X0[1] + 0.5*dy, 
         z = X0[2] + 0.5*dz;

  if (By()>0) { // Y-Z gauge
    gx = dx * Kex(); 
    gy =-dy * x * Bz(); // -dy*x^hat*Bz
    gz = dz * x * By(); //  dz*x^hat*By
  } else { // X-Z gauge
    gx = dx * y * Bz() + dx * Kex(); //  dx*y^hat*Bz + dx*K
    gy = 0; 
    gz =-dz * y * Bx(); // -dz*y^hat*Bx
  }

  return gx + gy + gz; 
}
