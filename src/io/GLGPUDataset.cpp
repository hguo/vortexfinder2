#include "GLGPUDataset.h"
#include "GLGPU_IO_Helper.h"
#include "common/Utils.hpp"
#include <cassert>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <glob.h>

#if WITH_PROTOBUF
#include "common/DataInfo.pb.h"
#endif

template <typename T>
void free1(T **p)
{
  if (*p != NULL) {
    free(*p);
    *p = NULL;
  }
}

GLGPUDataset::GLGPUDataset()
{
  memset(_rho, 0, sizeof(float*)*2);
  memset(_phi, 0, sizeof(float*)*2);
  memset(_re, 0, sizeof(float*)*2);
  memset(_im, 0, sizeof(float*)*2);
  memset(_J, 0, sizeof(float*)*2);
}

GLGPUDataset::~GLGPUDataset()
{
  for (int i=0; i<2; i++) {
    free1(&_rho[i]);
    free1(&_phi[i]);
    free1(&_re[i]);
    free1(&_im[i]);
    free1(&_J[i]);
  }
}

void GLGPUDataset::PrintInfo(int slot) const
{
  const GLHeader &h = _h[slot];

  fprintf(stderr, "dtype=%d\n", h.dtype);
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
#if WITH_PROTOBUF
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
#endif
}

bool GLGPUDataset::OpenDataFile(const std::string &filename)
{
  std::ifstream ifs;
  ifs.open(filename.c_str(), std::ifstream::in);
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

  // fprintf(stderr, "found %lu files\n", _filenames.size());
  return _filenames.size()>0;
}

void GLGPUDataset::CloseDataFile()
{
  _filenames.clear();
}

bool GLGPUDataset::LoadTimeStep(int timestep, int slot)
{
  assert(timestep>=0 && timestep<=_filenames.size());
  bool succ = false;
  const std::string &filename = _filenames[timestep];

  // load
  if (OpenBDATDataFile(filename, slot)) succ = true; 
  else if (OpenLegacyDataFile(filename, slot)) succ = true;

  if (!succ) return false;

  if (_precompute_supercurrent) 
    ComputeSupercurrentField(slot);

  // ModulateKex(slot);
  // fprintf(stderr, "loaded time step %d, %s\n", timestep, _filenames[timestep].c_str());

  SetTimeStep(timestep, slot);
  return true;
}

void GLGPUDataset::GetDataArray(GLHeader& h, float **rho, float **phi, float **re, float **im, float **J, int slot)
{
  h = _h[slot];
  *rho = _rho[slot];
  *phi = _phi[slot];
  *re = _re[slot]; 
  *im = _im[slot];
  *J = _J[slot];
}

bool GLGPUDataset::BuildDataFromArray(const GLHeader& h, const float *rho, const float *phi, const float *re, const float *im)
{
  memcpy(&_h[0], &h, sizeof(GLHeader));

  const int count = h.dims[0]*h.dims[1]*h.dims[2];
  // _psi[0] = (float*)realloc(_psi[0], sizeof(float)*count*2);
  _rho[0] = (float*)malloc(sizeof(float)*count); 
  _phi[0] = (float*)malloc(sizeof(float)*count); 
  _re[0] = (float*)malloc(sizeof(float)*count); 
  _im[0] = (float*)malloc(sizeof(float)*count); 

  memcpy(_rho[0], rho, sizeof(float)*count);
  memcpy(_phi[0], phi, sizeof(float)*count);
  memcpy(_re[0], re, sizeof(float)*count);
  memcpy(_im[0], im, sizeof(float)*count);
  
  return true;
}

#if 0
void GLGPUDataset::ModulateKex(int slot)
{
  float K = Kex(slot);
  float *re = slot == 0 ? _re : _re1,
         *im = slot == 0 ? _im : _im1;

  for (int i=0; i<dims()[0]; i++) 
    for (int j=0; j<dims()[1]; j++)
      for (int k=0; k<dims()[2]; k++) {
        const int idx[3] = {i, j, k};
        NodeIdType nid = Idx2Nid(idx);
        float x = i * CellLengths()[0] + Origins()[0];

        float rho = sqrt(re[nid]*re[nid] + im[nid]*im[nid]), 
               // phi = atan2(im[nid], re[nid]) - K*x;
               phi = atan2(im[nid], re[nid]) + K*x;

        re[nid] = rho * cos(phi);
        im[nid] = rho * sin(phi);
      }
}
#endif

void GLGPUDataset::RotateTimeSteps()
{
  std::swap(_rho[0], _rho[1]);
  std::swap(_phi[0], _phi[1]);
  std::swap(_re[0], _re[1]);
  std::swap(_im[0], _im[1]);
  std::swap(_J[0], _J[1]);

  GLDataset::RotateTimeSteps();
}

bool GLGPUDataset::OpenLegacyDataFile(const std::string& filename, int slot)
{
  int ndims;
  _h[slot].dtype = DTYPE_CA02;

  free1(&_rho[slot]); 
  free1(&_phi[slot]); 
  free1(&_re[slot]); 
  free1(&_im[slot]);
  free1(&_J[slot]);

  if (!::GLGPU_IO_Helper_ReadLegacy(
        filename, _h[slot], &_rho[slot], &_phi[slot], &_re[slot], &_im[slot]))
    return false;
  else 
    return true;
}

bool GLGPUDataset::OpenBDATDataFile(const std::string& filename, int slot)
{
  int ndims;
  _h[slot].dtype = DTYPE_BDAT;
  
  free1(&_rho[slot]); 
  free1(&_phi[slot]); 
  free1(&_re[slot]); 
  free1(&_im[slot]); 
  free1(&_J[slot]);

  if (!::GLGPU_IO_Helper_ReadBDAT(
        filename, _h[slot], &_rho[slot], &_phi[slot], &_re[slot], &_im[slot]))
    return false;
  else 
    return true;
}

#if 0
float Ax(const float X[3], int slot=0) const {if (By()>0) return -Kex(slot); else return -X[1]*Bz()-Kex(slot);}
// float Ax(const float X[3], int slot=0) const {if (By()>0) return 0; else return -X[1]*Bz();}
float Ay(const float X[3], int slot=0) const {if (By()>0) return X[0]*Bz(); else return 0;}
float Az(const float X[3], int slot=0) const {if (By()>0) return -X[0]*By(); else return X[1]*Bx();}
#endif

float GLGPUDataset::Rho(int i, int j, int k, int slot) const
{
  int idx[3] = {i, j, k};
  NodeIdType nid = Idx2Nid(idx);
  return Rho(nid, slot);
}

float GLGPUDataset::Phi(int i, int j, int k, int slot) const
{
  int idx[3] = {i, j, k};
  NodeIdType nid = Idx2Nid(idx);
  return Phi(nid, slot);
}

float GLGPUDataset::Re(int i, int j, int k, int slot) const
{
  int idx[3] = {i, j, k};
  NodeIdType nid = Idx2Nid(idx);
  return Re(nid, slot);
}

float GLGPUDataset::Im(int i, int j, int k, int slot) const
{
  int idx[3] = {i, j, k};
  NodeIdType nid = Idx2Nid(idx);
  return Im(nid, slot);
}

bool GLGPUDataset::A(const float X[3], float A[3], int slot) const
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

bool GLGPUDataset::A(NodeIdType n, float A_[3], int slot) const
{
  float X[3];
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

void GLGPUDataset::Idx2Pos(const int idx[], float pos[]) const
{
  for (int i=0; i<3; i++) 
    pos[i] = idx[i] * CellLengths()[i] + Origins()[i];
}

void GLGPUDataset::Pos2Idx(const float pos[], int idx[]) const
{
  for (int i=0; i<3; i++)
    idx[i] = (pos[i] - Origins()[i]) / CellLengths()[i]; 
  // TODO: perodic boundary conditions
}

void GLGPUDataset::Pos2Grid(const float pos[], float gpos[]) const
{
  for (int i=0; i<3; i++)
    gpos[i] = (pos[i] - Origins()[i]) / CellLengths()[i]; 
}

bool GLGPUDataset::Pos(NodeIdType id, float X[3]) const
{
  int idx[3];

  Nid2Idx(id, idx);
  Idx2Pos(idx, X);

  return true;
}

bool GLGPUDataset::Supercurrent(const float X[2], float J[3], int slot) const
{
  // TODO
  return false;
}

bool GLGPUDataset::Supercurrent(NodeIdType, float J[3], int slot) const
{
  // TODO
  return false;
}

#if 0
float GLGPUDataset::QP(const float X0[], const float X1[]) const 
{
  const float *L = Lengths(), 
               *O = Origins();
  float d[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  int p[3] = {0}; // 0: not crossed; 1: positive; -1: negative

  for (int i=0; i<3; i++) {
    d[i] = X1[i] - X0[i];
    if (d[i]>L[i]/2) {d[i] -= L[i]; p[i] = 1;}
    else if (d[i]<-L[i]/2) {d[i] += L[i]; p[i] = -1;}
  }

  const float X[3] = {X0[0] - O[0], X0[1] - O[1], X0[2] - O[2]};

  if (By()>0 && p[0]!=0) { // By>0
    return p[0] * L[0] * (Bz()*X[1] - By()*X[2]); 
  } else if (p[1]!=0) {
    return p[1] * L[1] * (Bx()*X[2] - Bz()*X[0]);
  } else return 0.0;
}
#else
float GLGPUDataset::QP(const float X0_[], const float X1_[], int slot) const
{
  float X0[3], X1[3];
  float N[3];
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
    float dj = X1[1] - X0[1];
    if (dj > N[1]/2) dj = dj - N[1];
    else if (dj < -N[1]/2) dj = dj + N[1];
    
    float dist = fabs(dj);
    float dist1 = fabs(fmod1(X0[1] + N[1]/2, N[1]) - N[1]);
    float f = dist1/dist;

    // pbc k
    float dk = X1[2] - X0[2];
    if (dk > N[2]/2) dk = dk - N[2];
    else if (dk < -N[2]/2) dk = dk + N[2];
    float k = fmod1(X0[2] + f*dk, N[2]);

    // pbc i
    float di = X1[0] - X0[0];
    if (di > N[0]/2) di = di - N[0];
    else if (di < -N[0]/2) di = di + N[0];
    float i = fmod1(X0[0] + f*dk, N[0]);

    float sign = dj>0 ? 1 : -1;
    float qp = sign * (k*CellLengths()[2]*B(slot)[0]*Lengths()[1] - i*CellLengths()[0]*B(slot)[2]*Lengths()[1]);

    return qp;
  } 
  
  return 0.0;
}
#endif
