#include <netcdf.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include "GLGPUDataset.h"

#define NC_SAFE_CALL(call) {\
  int retval = call;\
  if (retval != 0) {\
      fprintf(stderr, "[NetCDF Error] %s, in file '%s', line %i.\n", nc_strerror(retval), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  }\
}

static const int GLGPU_TAG_SIZE = 4;
static const char GLGPU_TAG[] = "CA02"; 

GLGPUDataset::GLGPUDataset()
{
  for (int i=0; i<3; i++) {
    _dims[i] = 1; 
    _pbc[i] = false;
  }
}

GLGPUDataset::~GLGPUDataset()
{
}

void GLGPUDataset::WriteToNetCDF(const std::string& filename)
{
#if 0
  int ncid; 
  int dimids[3]; 
  int varids[4];

  size_t starts[3] = {0, 0, 0}, 
         sizes[3]  = {_dims[2], _dims[1], _dims[0]};
         // sizes[3]  = {_dims[0], _dims[1], _dims[2]};

  NC_SAFE_CALL( nc_create(filename, NC_CLOBBER | NC_64BIT_OFFSET, &ncid) ); 
  NC_SAFE_CALL( nc_def_dim(ncid, "z", sizes[0], &dimids[0]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "y", sizes[1], &dimids[1]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "x", sizes[2], &dimids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "amp", NC_FLOAT, 3, dimids, &varids[0]) );
  NC_SAFE_CALL( nc_def_var(ncid, "phase", NC_FLOAT, 3, dimids, &varids[1]) );
  NC_SAFE_CALL( nc_def_var(ncid, "re", NC_FLOAT, 3, dimids, &varids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "im", NC_FLOAT, 3, dimids, &varids[3]) );
  NC_SAFE_CALL( nc_enddef(ncid) );

  NC_SAFE_CALL( nc_put_vara_float(ncid, varids[0], starts, sizes, _amp) ); 
  NC_SAFE_CALL( nc_put_vara_float(ncid, varids[1], starts, sizes, _phase) ); 
  NC_SAFE_CALL( nc_put_vara_float(ncid, varids[2], starts, sizes, _re) ); 
  NC_SAFE_CALL( nc_put_vara_float(ncid, varids[3], starts, sizes, __im) ); 

  NC_SAFE_CALL( nc_close(ncid) );
#endif
}

bool GLGPUDataset::LoadFromFile(const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;

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
    _cellLengths[i] = _lengths[i] / (_dims[i]-1); 
  }
  fprintf(stderr, "dims={%d, %d, %d}\n", _dims[0], _dims[1], _dims[2]); 
  fprintf(stderr, "lengths={%f, %f, %f}\n", _lengths[0], _lengths[1], _lengths[2]);
  fprintf(stderr, "cellLengths={%f, %f, %f}\n", _cellLengths[0], _cellLengths[1], _cellLengths[2]); 

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
    _Jx = Jx; 
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    double time; 
    fread(&time, sizeof(double), 1, fp); 
    fread(&_fluctuation_amp, sizeof(double), 1, fp);
    fread(_B, sizeof(double), 3, fp);
    fread(&_Jx, sizeof(double), GLGPU_TYPE_FLOAT, fp); 
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
