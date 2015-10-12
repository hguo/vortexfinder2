#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include "GLDataset.h"
#include "common/Utils.hpp"

GLDataset::GLDataset() : 
  _valid(false)
{
}

GLDataset::~GLDataset()
{
}

void GLDataset::RotateTimeSteps()
{
  GLDatasetBase::RotateTimeSteps();
}

bool GLDataset::OpenDataFile(const std::string& filename)
{
  // no impl
  return false;
}

void GLDataset::CloseDataFile()
{
  // no impl
}

double GLDataset::LineIntegral(const double X0[], const double X1[], const double A0[], const double A1[]) const
{
  double dX[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  double A[3] = {A0[0]+A1[0], A0[1]+A1[1], A0[2]+A1[2]};

  for (int i=0; i<3; i++)
    if (dX[i] > Lengths()[i]/2) dX[i] -= Lengths()[i];
    else if (dX[i] < -Lengths()[i]/2) dX[i] += Lengths()[i];

  return 0.5 * inner_product(A, dX);
}

double GLDataset::QP(const double X0[], const double X1[], int slot) const
{
  return 0.0;
}

#if 0
bool GLDataset::Rho(const double X[3], double &rho, int slot) const
{
  double re, im;
  bool succ = Psi(X, re, im, slot); 
  if (!succ) return false; 
  else {
    rho = sqrt(re*re + im*im); 
    return true;
  }
}

bool GLDataset::Phi(const double X[3], double &phi, int slot) const
{
  double re, im;
  bool succ = Psi(X, re, im, slot); 
  if (!succ) return false; 
  else {
    phi = atan2(im, re);
    return true;
  }
}
#endif

static void AverageA(int n, double A[][3])
{
  double AA[3] = {0};
  for (int i=0; i<3; i++) {
    for (int j=0; j<n; j++) 
      AA[i] += A[j][i];
    AA[i] /= n;
  }
}

void GLDataset::GetFaceValues(const CFace& f, int slot, double X[][3], double A_[][3], double rho[], double phi[], double re[], double im[]) const
{
  for (int i=0; i<f.nodes.size(); i++) {
    Pos(f.nodes[i], X[i]);
    A(f.nodes[i], A_[i], slot);
    // RhoPhi(f.nodes[i], rho[i], phi[i], slot);
    RhoPhiReIm(f.nodes[i], rho[i], phi[i], re[i], im[i], slot);
  }
    
  AverageA(f.nodes.size(), A_);
}

void GLDataset::GetSpaceTimeEdgeValues(const CEdge& e, double X[][3], double A_[][3], double rho[], double phi[], double re[], double im[]) const
{
  Pos(e.node0, X[0]);
  Pos(e.node1, X[1]);

  A(e.node0, A_[0], 0);
  A(e.node1, A_[1], 0);
  A(e.node1, A_[2], 1);
  A(e.node0, A_[3], 1);

  RhoPhiReIm(e.node0, rho[0], phi[0], re[0], im[0], 0);
  RhoPhiReIm(e.node1, rho[1], phi[1], re[1], im[1], 0);
  RhoPhiReIm(e.node1, rho[2], phi[2], re[2], im[2], 1);
  RhoPhiReIm(e.node0, rho[3], phi[3], re[3], im[3], 1);
}

