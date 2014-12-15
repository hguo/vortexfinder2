#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include "common/Utils.hpp"
#include "GLGPUDataset.h"
#include "common/DataInfo.pb.h"

#ifdef WITH_LIBMESH // suppose libmesh is built with netcdf
#include <netcdf.h>
#endif

static const int GLGPU_TAG_SIZE = 4;
static const char GLGPU_TAG[] = "CA02"; 

enum {
  GLGPU_ENDIAN_LITTLE = 0, 
  GLGPU_ENDIAN_BIG = 1
};

enum {
  GLGPU_TYPE_FLOAT = 0, 
  GLGPU_TYPE_DOUBLE = 1
};

GLGPUDataset::GLGPUDataset() : 
  _re(NULL), _im(NULL), _amp(NULL), _phase(NULL), 
  _scx(NULL), _scy(NULL), _scz(NULL), _scm(NULL)
{
  for (int i=0; i<3; i++) {
    _dims[i] = 1; 
    _pbc[i] = false;
  }
}

GLGPUDataset::~GLGPUDataset()
{
  if (_re) free(_re);
  if (_im) free(_im);
  if (_amp) free(_amp);
  if (_phase) free (_phase);
  if (_scx) free(_scx);
}

void GLGPUDataset::PrintInfo() const
{
  // TODO
}

void GLGPUDataset::ElemId2Idx(unsigned int id, int *idx) const
{
  int s = dims()[0] * dims()[1]; 
  int k = id / s; 
  int j = (id - k*s) / dims()[0]; 
  int i = id - k*s - j*dims()[0]; 

  idx[0] = i; idx[1] = j; idx[2] = k;
}

unsigned int GLGPUDataset::Idx2ElemId(int *idx) const
{
  for (int i=0; i<3; i++) 
    if (idx[i]<0 || idx[i]>=dims()[i])
      return UINT_MAX;
  
  return idx[0] + dims()[0] * (idx[1] + dims()[1] * idx[2]); 
}

void GLGPUDataset::Idx2Pos(const int idx[], double *pos) const
{
  for (int i=0; i<3; i++) 
    pos[i] = idx[i] * CellLengths()[i] + Origins()[i];
}

void GLGPUDataset::Pos2Id(const double pos[], int *idx) const
{
  for (int i=0; i<3; i++)
    idx[i] = (pos[i] - Origins()[i]) / CellLengths()[i]; 
  // TODO: perodic boundary conditions
}

double GLGPUDataset::Flux(int face) const
{
  // TODO: pre-compute the flux
  switch (face) {
  case 0: return -dx() * dy() * Bz();
  case 1: return -dy() * dz() * Bx(); 
  case 2: return -dz() * dx() * By(); 
  case 3: return  dx() * dy() * Bz(); 
  case 4: return  dy() * dz() * Bx(); 
  case 5: return  dz() * dx() * By();
  default: assert(false);
  }
}
  
double GLGPUDataset::GaugeTransformation(const int idx0[3], const int idx1[3]) const
{
  double X0[3], X1[3]; 

  Idx2Pos(idx0, X0); 
  Idx2Pos(idx1, X1); 
  
  return GLDataset::GaugeTransformation(X0, X1);
}

double GLGPUDataset::GaugeTransformation(int x0, int y0, int z0, int x1, int y1, int z1) const
{
  int idx0[3] = {x0, y0, z0}, 
      idx1[3] = {x1, y1, z1}; 

  return GaugeTransformation(idx0, idx1);
}

void GLGPUDataset::GetFace(int idx0[3], int face, int X[4][3]) const
{
  int idx1[3] = {(idx0[0]+1)%dims()[0], (idx0[1]+1)%dims()[1], (idx0[2]+1)%dims()[2]}; 
  
  switch (face) {
  case 0: // XY0
    X[0][0] = idx0[0]; X[0][1] = idx0[1]; X[0][2] = idx0[2]; 
    X[1][0] = idx0[0]; X[1][1] = idx1[1]; X[1][2] = idx0[2]; 
    X[2][0] = idx1[0]; X[2][1] = idx1[1]; X[2][2] = idx0[2]; 
    X[3][0] = idx1[0]; X[3][1] = idx0[1]; X[3][2] = idx0[2]; 
    break; 
  
  case 1: // YZ0
    X[0][0] = idx0[0]; X[0][1] = idx0[1]; X[0][2] = idx0[2]; 
    X[1][0] = idx0[0]; X[1][1] = idx0[1]; X[1][2] = idx1[2]; 
    X[2][0] = idx0[0]; X[2][1] = idx1[1]; X[2][2] = idx1[2]; 
    X[3][0] = idx0[0]; X[3][1] = idx1[1]; X[3][2] = idx0[2]; 
    break; 
  
  case 2: // ZX0
    X[0][0] = idx0[0]; X[0][1] = idx0[1]; X[0][2] = idx0[2]; 
    X[1][0] = idx1[0]; X[1][1] = idx0[1]; X[1][2] = idx0[2]; 
    X[2][0] = idx1[0]; X[2][1] = idx0[1]; X[2][2] = idx1[2]; 
    X[3][0] = idx0[0]; X[3][1] = idx0[1]; X[3][2] = idx1[2]; 
    break; 
  
  case 3: // XY1
    X[0][0] = idx0[0]; X[0][1] = idx0[1]; X[0][2] = idx1[2]; 
    X[1][0] = idx1[0]; X[1][1] = idx0[1]; X[1][2] = idx1[2]; 
    X[2][0] = idx1[0]; X[2][1] = idx1[1]; X[2][2] = idx1[2]; 
    X[3][0] = idx0[0]; X[3][1] = idx1[1]; X[3][2] = idx1[2]; 
    break; 

  case 4: // YZ1
    X[0][0] = idx1[0]; X[0][1] = idx0[1]; X[0][2] = idx0[2]; 
    X[1][0] = idx1[0]; X[1][1] = idx1[1]; X[1][2] = idx0[2]; 
    X[2][0] = idx1[0]; X[2][1] = idx1[1]; X[2][2] = idx1[2]; 
    X[3][0] = idx1[0]; X[3][1] = idx0[1]; X[3][2] = idx1[2]; 
    break; 

  case 5: // ZX1
    X[0][0] = idx0[0]; X[0][1] = idx1[1]; X[0][2] = idx0[2]; 
    X[1][0] = idx0[0]; X[1][1] = idx1[1]; X[1][2] = idx1[2]; 
    X[2][0] = idx1[0]; X[2][1] = idx1[1]; X[2][2] = idx1[2]; 
    X[3][0] = idx1[0]; X[3][1] = idx1[1]; X[3][2] = idx0[2]; 
    break; 

  default: assert(0); break;  
  }
}

std::vector<unsigned int> GLGPUDataset::Neighbors(unsigned int elem_id) const
{
  std::vector<unsigned int> neighbors; 

  int idx[3], idx1[3];
  ElemId2Idx(elem_id, idx); 

  for (int face=0; face<6; face++) {
    switch (face) {
    case 0: idx1[0] = idx[0]; idx1[1] = idx[1]; idx1[2] = idx[2]-1; break; 
    case 1: idx1[0] = idx[0]-1; idx1[1] = idx[1]; idx1[2] = idx[2]; break;
    case 2: idx1[0] = idx[0]; idx1[1] = idx[1]-1; idx1[2] = idx[2]; break;
    case 3: idx1[0] = idx[0]; idx1[1] = idx[1]; idx1[2] = idx[2]+1; break; 
    case 4: idx1[0] = idx[0]+1; idx1[1] = idx[1]; idx1[2] = idx[2]; break;
    case 5: idx1[0] = idx[0]; idx1[1] = idx[1]+1; idx1[2] = idx[2]; break;
    default: break;
    }

    for (int i=0; i<3; i++) 
      if (pbc()[i]) {
        idx1[i] = idx1[i] % dims()[i]; 
        if (idx1[i]<0) idx1[i] += dims()[i];
      }
    
    neighbors.push_back(Idx2ElemId(idx1)); 
  }

  return neighbors; 
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

  pb.set_bx(Bx());
  pb.set_by(By());
  pb.set_bz(Bz());

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
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;

  _data_name = filename;

  // tag check
  char tag[GLGPU_TAG_SIZE+1] = {0};  
  fread(tag, 1, GLGPU_TAG_SIZE, fp);
  if (strcmp(tag, GLGPU_TAG) != 0) return false;

  // endians
  int endian; 
  fread(&endian, sizeof(int), 1, fp); 

  // num_dims
  int num_dims; 
  fread(&num_dims, sizeof(int), 1, fp);
  fprintf(stderr, "num_dims=%d\n", num_dims); 

  // data type
  int size_real, datatype; 
  fread(&size_real, sizeof(int), 1, fp);
  if (size_real == 4) datatype = GLGPU_TYPE_FLOAT; 
  else if (size_real == 8) datatype = GLGPU_TYPE_DOUBLE; 
  else assert(false); 

  // dimensions 
  for (int i=0; i<num_dims; i++) {
    fread(&_dims[i], sizeof(int), 1, fp);
    if (datatype == GLGPU_TYPE_FLOAT) {
      float length; 
      fread(&length, sizeof(float), 1, fp);
      _lengths[i] = length; 
    } else if (datatype == GLGPU_TYPE_DOUBLE) {
      fread(&_lengths[i], sizeof(double), 1, fp); 
    }
    _origins[i] = -0.5*_lengths[i];
  }
  fprintf(stderr, "dims={%d, %d, %d}\n", _dims[0], _dims[1], _dims[2]); 
  fprintf(stderr, "origins={%f, %f, %f}\n", _origins[0], _origins[1], _origins[2]);
  fprintf(stderr, "lengths={%f, %f, %f}\n", _lengths[0], _lengths[1], _lengths[2]);

  // dummy
  int dummy; 
  fread(&dummy, sizeof(int), 1, fp);

  // time, fluctuation_amp, Bx, By, Bz, Jx
  if (datatype == GLGPU_TYPE_FLOAT) {
    float time, fluctuation_amp, B[3], Jx; 
    fread(&time, sizeof(float), 1, fp);
    fread(&fluctuation_amp, sizeof(float), 1, fp); 
    fread(&B, sizeof(float), 3, fp);
    fread(&Jx, sizeof(float), 1, fp); 
    // _time = time; 
    _fluctuation_amp = fluctuation_amp;
    _B[0] = B[0]; _B[1] = B[1]; _B[2] = B[2];
    // _Jx = Jx; 
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    double time, Jx;  
    fread(&time, sizeof(double), 1, fp); 
    fread(&_fluctuation_amp, sizeof(double), 1, fp);
    fread(_B, sizeof(double), 3, fp);
    fread(&Jx, sizeof(double), GLGPU_TYPE_FLOAT, fp); 
  }
    
  // fprintf(stderr, "time=%f\n", time); 
  fprintf(stderr, "fluctuation_amp=%f\n", _fluctuation_amp); 
  fprintf(stderr, "B={%f, %f, %f}\n", _B[0], _B[1], _B[2]); 

  // btype
  int btype; 
  fread(&btype, sizeof(int), 1, fp); 
  _pbc[0] = btype & 0x0000ff;
  _pbc[1] = btype & 0x00ff00;
  _pbc[2] = btype & 0xff0000; 
  fprintf(stderr, "pbc={%d, %d, %d}\n", _pbc[0], _pbc[1], _pbc[2]); 
  // update cell lengths 
  for (int i=0; i<num_dims; i++) 
    if (_pbc[i]) _cell_lengths[i] = _lengths[i] / _dims[i];  
    else _cell_lengths[i] = _lengths[i] / (_dims[i]-1); 
  fprintf(stderr, "cell_lengths={%f, %f, %f}\n", _cell_lengths[0], _cell_lengths[1], _cell_lengths[2]); 

  // optype
  int optype; 
  fread(&optype, sizeof(int), 1, fp);
  fprintf(stderr, "optype=%d\n", optype); 
  if (datatype == GLGPU_TYPE_FLOAT) {
    float Kex, Kex_dot; 
    fread(&Kex, sizeof(float), 1, fp);
    fread(&Kex_dot, sizeof(float), 1, fp); 
    _Kex = Kex; 
    _Kex_dot = Kex_dot; 
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    fread(&_Kex, sizeof(double), 1, fp);
    fread(&_Kex_dot, sizeof(double), 1, fp); 
  }
  fprintf(stderr, "Kex=%f, Kex_dot=%f\n", _Kex, _Kex_dot); 

  int count = 1; 
  for (int i=0; i<num_dims; i++) 
    count *= _dims[i]; 

  int offset = ftell(fp);
  fprintf(stderr, "offset=%d\n", offset); 
  
  // mem allocation 
  _re = (double*)malloc(sizeof(double)*count);  
  _im = (double*)malloc(sizeof(double)*count);
  _amp = (double*)malloc(sizeof(double)*count);
  _phase = (double*)malloc(sizeof(double)*count); 

  if (datatype == GLGPU_TYPE_FLOAT) {
    // raw data
    float *buf = (float*)malloc(sizeof(float)*count*2); // complex numbers
    fread(buf, sizeof(float), count*2, fp);

    // separation of ch1 and ch2
    float *ch1 = (float*)malloc(sizeof(float)*count), 
          *ch2 = (float*)malloc(sizeof(float)*count);
    for (int i=0; i<count; i++) {
      ch1[i] = buf[i*2]; 
      ch2[i] = buf[i*2+1];
    }
    free(buf); 

    if (optype == 0) { // order parameter type
      for (int i=0; i<count; i++) {
        _re[i] = ch1[i]; 
        _im[i] = ch2[i]; 
        _amp[i] = sqrt(_re[i]*_re[i] + _im[i]*_im[i]);
        _phase[i] = atan2(_im[i], _re[i]); 
        // fprintf(stderr, "amp=%f, phase=%f, re=%f, im=%f\n", _amp[i], _phase[i], _re[i], _im[i]); 
      }
    } else if (optype == 1) {
      for (int i=0; i<count; i++) {
        _amp[i] = ch1[i]; 
        _phase[i] = ch2[i]; 
        _re[i] = _amp[i] * cos(_phase[i]); 
        _im[i] = _amp[i] * sin(_phase[i]);
      }
    } else assert(false); 
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    assert(false);
    // The following lines are copied from legacy code. To be reorganized later
#if 0
    // raw data
    double *buf = (double*)malloc(sizeof(double)*count*2); // complex
    fread(buf, sizeof(double), count, fp);

    // separation of ch1 and ch2
    double *ct1 = (double*)malloc(sizeof(double)*count), 
           *ct2 = (double*)malloc(sizeof(double)*count);
    for (int i=0; i<count; i++) {
      ct1[i] = buf[i*2]; 
      ct2[i] = buf[i*2+1];
    }
    free(buf); 

    // transpose
    double *ch1 = (double*)malloc(sizeof(double)*count), 
           *ch2 = (double*)malloc(sizeof(double)*count);
    int dims1[] = {_dims[2], _dims[1], _dims[0]}; 
    for (int i=0; i<_dims[0]; i++) 
      for (int j=0; j<_dims[1]; j++) 
        for (int k=0; k<_dims[2]; k++) {
          texel3D(ch1, _dims, i, j, k) = texel3D(ct1, dims1, k, j, i);
          texel3D(ch2, _dims, i, j, k) = texel3D(ct2, dims1, k, j, i); 
        }

    if (optype == 0) {
      _re = ch1; 
      _im = ch2;
      _amp = (double*)malloc(sizeof(double)*count);
      _phase = (double*)malloc(sizeof(double)*count); 
      for (int i=0; i<count; i++) {
        _amp[i] = sqrt(_re[i]*_re[i] + _im[i]*_im[i]);
        _phase[i] = atan2(_im[i], _re[i]); 
      }
    } else if (optype == 1) {
      _amp = ch1; 
      _phase = ch2;
      _re = (double*)malloc(sizeof(double)*count); 
      _im = (double*)malloc(sizeof(double)*count); 
      for (int i=0; i<count; i++) {
        _re[i] = _amp[i] * cos(_phase[i]); 
        _im[i] = _amp[i] * sin(_phase[i]); 
      }
    } else assert(false);
#endif
  }
  
  return true; 
}

void GLGPUDataset::ComputeSupercurrentField()
{
  const int nvoxels = dims()[0]*dims()[1]*dims()[2];

  if (_scx != NULL) free(_scx);
  _scx = (double*)malloc(4*sizeof(double)*nvoxels);
  _scy = _scx + nvoxels; 
  _scz = _scy + nvoxels;
  _scm = _scz + nvoxels;
  memset(_scx, 0, 3*sizeof(double)*nvoxels);
 
  double dphi[3], sc[3], A[3];

  // central difference
  for (int x=0; x<dims()[0]; x++) {
    for (int y=0; y<dims()[1]; y++) {
      for (int z=0; z<dims()[2]; z++) {
        int idx[3] = {x, y, z}; 
        double pos[3]; 
        Idx2Pos(idx, pos);

        // boundaries. TODO: pbc
        int xp = std::max(0, x-1), 
            xq = std::min(dims()[0]-1, x+1), 
            yp = std::max(0, y-1), 
            yq = std::min(dims()[1]-1, y+1), 
            zp = std::max(0, z-1), 
            zq = std::min(dims()[2]-1, z+1); 

        // Q: should I do gauge transformation here?
#if 0
        dphi[0] = 0.5 * (mod2pi(phase(xq, y, z) - phase(xp, y, z) + GaugeTransformation(xq, y, z, xp, y, z) + M_PI) - M_PI) / dx();
        dphi[1] = 0.5 * (mod2pi(phase(x, yq, z) - phase(x, yp, z) + GaugeTransformation(x, yq, z, x, yp, z) + M_PI) - M_PI) / dy();
        dphi[2] = 0.5 * (mod2pi(phase(x, y, zq) - phase(x, y, zp) + GaugeTransformation(x, y, zq, x, y, zp) + M_PI) - M_PI) / dz();
#else
        dphi[0] = 0.5 * (mod2pi(phase(xq, y, z) - phase(xp, y, z) + M_PI) - M_PI) / dx();
        dphi[1] = 0.5 * (mod2pi(phase(x, yq, z) - phase(x, yp, z) + M_PI) - M_PI) / dy();
        dphi[2] = 0.5 * (mod2pi(phase(x, y, zq) - phase(x, y, zp) + M_PI) - M_PI) / dz();
#endif

        sc[0] = dphi[0] - Ax(pos);
        sc[1] = dphi[1] - Ax(pos);
        sc[2] = dphi[2] - Ax(pos);

        texel3D(_scx, dims(), x, y, z) = sc[0]; 
        texel3D(_scy, dims(), x, y, z) = sc[1];
        texel3D(_scz, dims(), x, y, z) = sc[2];
        texel3D(_scm, dims(), x, y, z) = sqrt(sc[0]*sc[0] + sc[1]*sc[1] + sc[2]*sc[2]);
      }
    }
  }
}

void GLGPUDataset::WriteNetCDFFile(const std::string& filename)
{
#ifdef WITH_LIBMESH
  int ncid; 
  int dimids[3]; 
  int varids[8];

  size_t starts[3] = {0, 0, 0}, 
         sizes[3]  = {_dims[2], _dims[1], _dims[0]};

  NC_SAFE_CALL( nc_create(filename.c_str(), NC_CLOBBER | NC_64BIT_OFFSET, &ncid) ); 
  NC_SAFE_CALL( nc_def_dim(ncid, "z", sizes[0], &dimids[0]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "y", sizes[1], &dimids[1]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "x", sizes[2], &dimids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "amp", NC_DOUBLE, 3, dimids, &varids[0]) );
  NC_SAFE_CALL( nc_def_var(ncid, "phase", NC_DOUBLE, 3, dimids, &varids[1]) );
  NC_SAFE_CALL( nc_def_var(ncid, "re", NC_DOUBLE, 3, dimids, &varids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "im", NC_DOUBLE, 3, dimids, &varids[3]) );
  NC_SAFE_CALL( nc_def_var(ncid, "scx", NC_DOUBLE, 3, dimids, &varids[4]) );
  NC_SAFE_CALL( nc_def_var(ncid, "scy", NC_DOUBLE, 3, dimids, &varids[5]) );
  NC_SAFE_CALL( nc_def_var(ncid, "scz", NC_DOUBLE, 3, dimids, &varids[6]) );
  NC_SAFE_CALL( nc_def_var(ncid, "scm", NC_DOUBLE, 3, dimids, &varids[7]) );
  NC_SAFE_CALL( nc_enddef(ncid) );

  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[0], starts, sizes, _amp) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[1], starts, sizes, _phase) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[2], starts, sizes, _re) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[3], starts, sizes, _im) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[4], starts, sizes, _scx) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[5], starts, sizes, _scy) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[6], starts, sizes, _scz) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[7], starts, sizes, _scm) ); 

  NC_SAFE_CALL( nc_close(ncid) );
#else
  assert(false);
#endif
}
