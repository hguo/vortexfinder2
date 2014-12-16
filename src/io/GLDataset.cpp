#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include "gldataset.h"
#include "common/Utils.hpp"

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
  
double GLDataset::GaugeTransformation(const double X0[], const double X1[]) const
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

double GLDataset::Flux(const double X[3][3]) const
{
  double A[3] = {X[1][0] - X[0][0], X[1][1] - X[0][1], X[1][2] - X[0][2]}, 
         B[3] = {X[2][0] - X[0][0], X[2][1] - X[0][1], X[2][2] - X[0][2]};
  double dS[3];

  cross_product(A, B, dS);

  return inner_product(_B, dS);
}

void GLDataset::A(const double X[3], double A[3]) const
{
  if (By()>0) {
    A[0] = 0;
    A[1] = X[0] * Bz(); 
    A[2] =-X[0] * By();
  } else {
    A[0] =-X[1] * Bz(); 
    A[1] = 0; 
    A[2] = X[1] * Bx();
  }
}
  
bool GLDataset::Rho(const double X[3], double &rho) const
{
  double re, im;
  bool succ = Psi(X, re, im); 
  if (!succ) return false; 
  else {
    rho = sqrt(re*re + im*im); 
    return true;
  }
}

bool GLDataset::Phi(const double X[3], double &phi) const
{
  double re, im;
  bool succ = Psi(X, re, im); 
  if (!succ) return false; 
  else {
    phi = atan2(im, re);
    return true;
  }
}
