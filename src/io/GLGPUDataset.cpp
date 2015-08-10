#include "GLGPUDataset.h"
#include "GLGPU_IO_Helper.h"
#include "common/Utils.hpp"
#include "common/DataInfo.pb.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
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

void GLGPUDataset::PrintInfo(int slot) const
{
  const GLHeader &h = _h[slot];

  fprintf(stderr, "dims={%d, %d, %d}\n", h.dims[0], h.dims[1], h.dims[2]); 
  fprintf(stderr, "pbc={%d, %d, %d}\n", h.pbc[0], h.pbc[1], h.pbc[2]); 
  fprintf(stderr, "origins={%f, %f, %f}\n", h.origins[0], h.origins[1], h.origins[2]);
  fprintf(stderr, "lengths={%f, %f, %f}\n", h.lengths[0], h.lengths[1], h.lengths[2]);
  fprintf(stderr, "cell_lengths={%f, %f, %f}\n", h.cell_lengths[0], h.cell_lengths[1], h.cell_lengths[2]); 
  fprintf(stderr, "B={%f, %f, %f}\n", h.B[0], h.B[1], h.B[2]);
  fprintf(stderr, "Kex=%f\n", h.Kex);
  fprintf(stderr, "Jxext=%f\n", h.Jxext);
  fprintf(stderr, "V=%f\n", h.V);
  fprintf(stderr, "time=%f\n", h.time);
  fprintf(stderr, "fluctuation_amp=%f\n", h.fluctuation_amp); 
}

void GLGPUDataset::SerializeDataInfoToString(std::string& buf) const
{
  PBDataInfo pb;

  pb.set_model(PBDataInfo::GLGPU);
  pb.set_name(_data_name);

  if (Lengths()[0]>0) {
    pb.set_ox(Origins()[0]); 
    pb.set_oy(Origins()[1]); 
    pb.set_oz(Origins()[2]); 
    pb.set_lx(Lengths()[0]); 
    pb.set_ly(Lengths()[1]); 
    pb.set_lz(Lengths()[2]); 
  }

  pb.set_bx(B()[0]);
  pb.set_by(B()[1]);
  pb.set_bz(B()[2]);

  pb.set_kex(Kex());

  pb.set_dx(dims()[0]);
  pb.set_dy(dims()[1]);
  pb.set_dz(dims()[2]);

  pb.set_pbc_x(pbc()[0]);
  pb.set_pbc_y(pbc()[0]);
  pb.set_pbc_z(pbc()[0]);

  pb.SerializeToString(&buf);
}

bool GLGPUDataset::OpenDataFile(const std::string &filename)
{
  std::ifstream ifs;
  ifs.open(filename, std::ifstream::in);
  if (!ifs.is_open()) return false;

  char fname[1024];

  _filenames.clear();
  while (ifs.getline(fname, 1024)) {
    // std::cout << fname << std::endl;
    _filenames.push_back(fname);
  }

  ifs.close();

  _data_name = filename;
  return true;
}

bool GLGPUDataset::OpenDataFileByPattern(const std::string &pattern)
{
  glob_t results;

  glob(pattern.c_str(), 0, NULL, &results);
  _filenames.clear();
  for (int i=0; i<results.gl_pathc; i++) 
    _filenames.push_back(results.gl_pathv[i]);

  fprintf(stderr, "found %lu files\n", _filenames.size());
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

  fprintf(stderr, "loading time step %d, %s\n", timestep, _filenames[timestep].c_str());

  // load
  if (OpenBDATDataFile(filename, slot)) succ = true; 
  else if (OpenLegacyDataFile(filename, slot)) succ = true;

  if (!succ) return;
#if 0 
  for (int i=0; i<Dimensions(); i++) {
    _origins[i] = -0.5*_lengths[i];
    if (_pbc[i]) _cell_lengths[i] = _lengths[i] / _dims[i];  
    else _cell_lengths[i] = _lengths[i] / (_dims[i]-1); 
  }
#endif

  // ModulateKex(slot);

  SetTimeStep(timestep, slot);
}

#if 0
bool GLGPUDataset::BuildDataFromArray(
      int ndims, 
      const int *dims, 
      const double *lengths,
      const bool *pbc,
      double time,
      const double *B,
      double Jxext, 
      double Kx, 
      double V,
      const double *re,
      const double *im)
{
  memcpy(_dims, dims, sizeof(int)*3);
  memcpy(_lengths, lengths, sizeof(double)*3);
  memcpy(_pbc, pbc, sizeof(bool)*3);
  _time = time;
  memcpy(_B, B, sizeof(double)*3);
  _Jxext = Jxext;
  _Kex = Kx; 
  _V = V;
  
  for (int i=0; i<Dimensions(); i++) {
    _origins[i] = -0.5*_lengths[i];
    if (_pbc[i]) _cell_lengths[i] = _lengths[i] / _dims[i];  
    else _cell_lengths[i] = _lengths[i] / (_dims[i]-1); 
  }

  int count = _dims[0]*_dims[1]*_dims[2];
  _re = (double*)malloc(sizeof(double)*count);
  _im = (double*)malloc(sizeof(double)*count);
  memcpy(_re, re, sizeof(double)*count);
  memcpy(_im, im, sizeof(double)*count);

  return true;
}
#endif

void GLGPUDataset::ModulateKex(int slot)
{
  double K = Kex(slot);
  double *re = slot == 0 ? _re : _re1,
         *im = slot == 0 ? _im : _im1;

  for (int i=0; i<dims()[0]; i++) 
    for (int j=0; j<dims()[1]; j++)
      for (int k=0; k<dims()[2]; k++) {
        const int idx[3] = {i, j, k};
        NodeIdType nid = Idx2Nid(idx);
        double x = i * CellLengths()[0] + Origins()[0];

        double rho = sqrt(re[nid]*re[nid] + im[nid]*im[nid]), 
               // phi = atan2(im[nid], re[nid]) - K*x;
               phi = atan2(im[nid], re[nid]) + K*x;

        re[nid] = rho * cos(phi);
        im[nid] = rho * sin(phi);
      }
}

void GLGPUDataset::RotateTimeSteps()
{
  double *r = _re, *i = _im;
  _re = _re1; _im = _im1;
  _re1 = r; _im1 = i;

  GLDataset::RotateTimeSteps();
}

bool GLGPUDataset::OpenLegacyDataFile(const std::string& filename, int slot)
{
  int ndims;
  if (!::GLGPU_IO_Helper_ReadLegacy(
        filename, _h[slot], 
        slot == 0 ? &_re : &_re1, 
        slot == 0 ? &_im : &_im1))
    return false;
  else 
    return true;
}

bool GLGPUDataset::OpenBDATDataFile(const std::string& filename, int slot)
{
  int ndims;
  if (!::GLGPU_IO_Helper_ReadBDAT(
        filename, _h[slot], 
        slot == 0 ? &_re : &_re1, 
        slot == 0 ? &_im : &_im1))
    return false;
  else 
    return true;
}

#if 0
double Ax(const double X[3], int slot=0) const {if (By()>0) return -Kex(slot); else return -X[1]*Bz()-Kex(slot);}
// double Ax(const double X[3], int slot=0) const {if (By()>0) return 0; else return -X[1]*Bz();}
double Ay(const double X[3], int slot=0) const {if (By()>0) return X[0]*Bz(); else return 0;}
double Az(const double X[3], int slot=0) const {if (By()>0) return -X[0]*By(); else return X[1]*Bx();}
#endif

bool GLGPUDataset::A(const double X[3], double A[3], int slot) const
{
  if (B(slot)[1]>0) {
    A[0] = -Kex(slot);
    A[1] = X[0] * B(slot)[2];
    A[2] = -X[0] * B(slot)[1];
  } else {
    A[0] = -X[1] * B(slot)[2] - Kex(slot);
    A[1] = 0;
    A[2] = X[1] * B(slot)[0];
  }
  
  return true;
}

bool GLGPUDataset::A(NodeIdType n, double A_[3], int slot) const
{
  double X[3];
  Pos(n, X);
  return A(X, A_, slot);
}

void GLGPUDataset::Nid2Idx(NodeIdType id, int *idx) const
{
  int s = dims()[0] * dims()[1]; 
  int k = id / s; 
  int j = (id - k*s) / dims()[0]; 
  int i = id - k*s - j*dims()[0]; 

  idx[0] = i; idx[1] = j; idx[2] = k;
}

NodeIdType GLGPUDataset::Idx2Nid(const int *idx) const
{
  for (int i=0; i<3; i++) 
    if (idx[i]<0 || idx[i]>=dims()[i])
      return UINT_MAX;
  
  return idx[0] + dims()[0] * (idx[1] + dims()[1] * idx[2]); 
}

void GLGPUDataset::Idx2Pos(const int idx[], double pos[]) const
{
  for (int i=0; i<3; i++) 
    pos[i] = idx[i] * CellLengths()[i] + Origins()[i];
}

void GLGPUDataset::Pos2Idx(const double pos[], int idx[]) const
{
  for (int i=0; i<3; i++)
    idx[i] = (pos[i] - Origins()[i]) / CellLengths()[i]; 
  // TODO: perodic boundary conditions
}

void GLGPUDataset::Pos2Grid(const double pos[], double gpos[]) const
{
  for (int i=0; i<3; i++)
    gpos[i] = (pos[i] - Origins()[i]) / CellLengths()[i]; 
}

bool GLGPUDataset::Pos(NodeIdType id, double X[3]) const
{
  int idx[3];

  Nid2Idx(id, idx);
  Idx2Pos(idx, X);

  return true;
}

bool GLGPUDataset::Psi(const double X[3], double &re, double &im, int slot) const
{
  // TODO
  return false;
}

bool GLGPUDataset::Psi(NodeIdType id, double &re, double &im, int slot) const
{
  double *r = slot == 0 ? _re : _re1;
  double *i = slot == 0 ? _im : _im1;

  re = r[id]; 
  im = i[id];

  return true;
}

bool GLGPUDataset::Supercurrent(const double X[2], double J[3], int slot) const
{
  // TODO
  return false;
}

bool GLGPUDataset::Supercurrent(NodeIdType, double J[3], int slot) const
{
  // TODO
  return false;
}

#if 0
double GLGPUDataset::QP(const double X0[], const double X1[]) const 
{
  const double *L = Lengths(), 
               *O = Origins();
  double d[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  int p[3] = {0}; // 0: not crossed; 1: positive; -1: negative

  for (int i=0; i<3; i++) {
    d[i] = X1[i] - X0[i];
    if (d[i]>L[i]/2) {d[i] -= L[i]; p[i] = 1;}
    else if (d[i]<-L[i]/2) {d[i] += L[i]; p[i] = -1;}
  }

  const double X[3] = {X0[0] - O[0], X0[1] - O[1], X0[2] - O[2]};

  if (By()>0 && p[0]!=0) { // By>0
    return p[0] * L[0] * (Bz()*X[1] - By()*X[2]); 
  } else if (p[1]!=0) {
    return p[1] * L[1] * (Bx()*X[2] - Bz()*X[0]);
  } else return 0.0;
}
#else
double GLGPUDataset::QP(const double X0_[], const double X1_[], int slot) const
{
  double X0[3], X1[3];
  double N[3];
  for (int i=0; i<3; i++) {
    X0[i] = (X0_[i] - Origins()[i]) / CellLengths()[i];
    X1[i] = (X1_[i] - Origins()[i]) / CellLengths()[i];
    N[i] = dims()[i];
  }

  if (B(slot)[1]>0 && fabs(X1[0]-X0[0])>N[0]/2) {
    // TODO
    assert(false);
    return 0.0;
  } else if (fabs(X1[1]-X0[1])>N[1]/2) {
    // pbc j
    double dj = X1[1] - X0[1];
    if (dj > N[1]/2) dj = dj - N[1];
    else if (dj < -N[1]/2) dj = dj + N[1];
    
    double dist = fabs(dj);
    double dist1 = fabs(fmod1(X0[1] + N[1]/2, N[1]) - N[1]);
    double f = dist1/dist;

    // pbc k
    double dk = X1[2] - X0[2];
    if (dk > N[2]/2) dk = dk - N[2];
    else if (dk < -N[2]/2) dk = dk + N[2];
    double k = fmod1(X0[2] + f*dk, N[2]);

    // pbc i
    double di = X1[0] - X0[0];
    if (di > N[0]/2) di = di - N[0];
    else if (di < -N[0]/2) di = di + N[0];
    double i = fmod1(X0[0] + f*dk, N[0]);

    double sign = dj>0 ? 1 : -1;
    double qp = sign * (k*CellLengths()[2]*B(slot)[0]*Lengths()[1] - i*CellLengths()[0]*B(slot)[2]*Lengths()[1]);

    return qp;
  } 
  
  return 0.0;
}
#endif
