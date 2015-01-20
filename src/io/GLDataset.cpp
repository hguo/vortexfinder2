#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include "gldataset.h"
#include "common/Utils.hpp"

GLDataset::GLDataset() : 
  _Kex(0), _Kex_dot(0), _fluctuation_amp(0), 
  _V(0), _Jx(0),
  _timestep(0), 
  _valid(false)
{
  memset(_origins, 0, sizeof(double)*3); 
  memset(_lengths, 0, sizeof(double)*3); 
}

GLDataset::~GLDataset()
{
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

double GLDataset::GaugeTransformation(const double X0[], const double X1[], const double A0[], const double A1[]) const
{
  // \int dl * (Kx^hat - A(l))

  double dX[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  double A[3] = {0.5*(A0[0]+A1[0]), 0.5*(A0[1]+A1[1]), 0.5*(A0[2]+A1[2])};

  double gl = Kex() * dX[0];
  double ga = -inner_product(A, dX);

  return gl + ga;
}

double GLDataset::LineIntegral(const double X0[], const double X1[], const double A0[], const double A1[]) const
{
  double dX[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  double A[3] = {0.5*(A0[0]+A1[0]), 0.5*(A0[1]+A1[1]), 0.5*(A0[2]+A1[2])};

  return inner_product(A, dX);
}

double GLDataset::QP(const double X0[], const double X1[]) const
{
  return 0.0;
}

#if 0
double GLDataset::Flux(const double X[3][3]) const
{
  double A[3] = {X[1][0] - X[0][0], X[1][1] - X[0][1], X[1][2] - X[0][2]}, 
         B[3] = {X[2][0] - X[0][0], X[2][1] - X[0][1], X[2][2] - X[0][2]};
  double dS[3];

  cross_product(A, B, dS);

  return inner_product(_B, dS);
}

double GLDataset::Flux(int n, const double X[][3]) const
{
  double flux = 0;
  for (int i=0; i<n-2; i++) {
    double X1[3][3] = {{X[0][0], X[0][1], X[0][2]}, 
                       {X[i+1][0], X[i+1][1], X[i+1][2]}, 
                       {X[i+2][0], X[i+2][1], X[i+2][2]}};
    flux += Flux(X1);
  }
  
  return flux;
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
#endif 

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

bool GLDataset::GetSpaceTimePrism(ElemIdType id, int face, double X[][3], 
      double A0[][3], double A1[][3], 
      double re0[], double re1[],
      double im0[], double im1[]) const
{
  return false;
}
