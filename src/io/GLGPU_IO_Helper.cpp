#include "GLGPU_IO_Helper.h"
#include <cmath>
#include <cassert>
#include <cstring>

#ifdef WITH_LIBMESH // suppose libmesh is built with netcdf
#include <netcdf.h>
#endif

enum {
  GLGPU_ENDIAN_LITTLE = 0, 
  GLGPU_ENDIAN_BIG = 1
};

enum {
  GLGPU_TYPE_FLOAT = 0, 
  GLGPU_TYPE_DOUBLE = 1
};

static const int GLGPU_LEGACY_TAG_SIZE = 4;
static const char GLGPU_LEGACY_TAG[] = "CA02";

bool GLGPU_IO_Helper_ReadBDAT(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double &time,
    double *B,
    double &Jxext, 
    double &Kex, 
    double &V, 
    double **re, 
    double **im)
{
  BDATReader *reader = new BDATReader(filename); 
  if (!reader->Valid()) {
    delete reader;
    return false;
  }

  std::string name, buf;
  while (1) {
    name = reader->ReadNextRecordInfo();
    if (name.size()==0) break;
    
    unsigned int type = reader->RecType(), 
                 recID = reader->RedID(); 
    float f; // temp var
    
    reader->ReadNextRecordData(&buf);
    void *p = (void*)buf.data();

    if (name == "dim") {
      assert(type == BDAT_INT32);
      memcpy(&ndims, p, sizeof(int));
      assert(ndims == 2 || ndims == 3);
    } else if (name == "Nx") {
      assert(type == BDAT_INT32);
      memcpy(&dims[0], p, sizeof(int));
    } else if (name == "Ny") {
      assert(type == BDAT_INT32);
      memcpy(&dims[1], p, sizeof(int));
    } else if (name == "Nz") {
      assert(type == BDAT_INT32);
      memcpy(&dims[2], p, sizeof(int));
    } else if (name == "Lx") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      lengths[0] = f;
    } else if (name == "Ly") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      lengths[1] = f;
    } else if (name == "Lz") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      lengths[2] = f;
    } else if (name == "BC") {
      assert(type == BDAT_INT32);
      int btype; 
      memcpy(&btype, p, sizeof(int));
      pbc[0] = ((btype & 0x0000ff) == 0x01);
      pbc[1] = ((btype & 0x00ff00) == 0x0100);
      pbc[2] = ((btype & 0xff0000) == 0x010000); 
    } else if (name == "zaniso") {
      assert(type == BDAT_FLOAT);
    } else if (name == "t") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      time = f;
    } else if (name == "Tf") {
      assert(type == BDAT_FLOAT);
    } else if (name == "Bx") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      B[0] = f;
    } else if (name == "By") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      B[1] = f;
    } else if (name == "Bz") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      B[2] = f;
    } else if (name == "Jxext") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      Jxext = f;
    } else if (name == "K") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      Kex = f;
    } else if (name == "V") {
      assert(type == BDAT_FLOAT);
      memcpy(&f, p, sizeof(float));
      V = f;
    } else if (name == "psi") {
      if (type == BDAT_FLOAT) {
        int count = buf.size()/sizeof(float)/2;
        int optype = recID == 2000 ? 0 : 1;
        float *data = (float*)p;
        
        *re = (double*)realloc(*re, sizeof(double)*count);
        *im = (double*)realloc(*im, sizeof(double)*count);

        if (optype == 0) { // re, im
          for (int i=0; i<count; i++) {
            (*re)[i] = data[i*2];
            (*im)[i] = data[i*2+1];
          }
        } else { // rho^2, phi
          for (int i=0; i<count; i++) {
            double rho = sqrt(data[i*2]), 
                   phi = data[i*2+1];
            (*re)[i] = rho * cos(phi); 
            (*im)[i] = rho * sin(phi);
            // fprintf(stderr, "rho=%f, phi=%f\n", rho, phi);
          }
        }
      } else if (type == BDAT_DOUBLE) {
        // TODO
        assert(false);
      } else 
        assert(false);
    }
  }

  delete reader;
  return true;
}

bool GLGPU_IO_Helper_ReadLegacy(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double &time,
    double *B,
    double &Jxext, 
    double &Kex, 
    double &V, 
    double **re, 
    double **im)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) return false;

  // tag check
  char tag[GLGPU_LEGACY_TAG_SIZE+1] = {0};  
  fread(tag, 1, GLGPU_LEGACY_TAG_SIZE, fp);
  if (strcmp(tag, GLGPU_LEGACY_TAG) != 0) return false;

  // endians
  int endian; 
  fread(&endian, sizeof(int), 1, fp); 

  // num_dims
  fread(&ndims, sizeof(int), 1, fp);

  // data type
  int size_real, datatype; 
  fread(&size_real, sizeof(int), 1, fp);
  if (size_real == 4) datatype = GLGPU_TYPE_FLOAT; 
  else if (size_real == 8) datatype = GLGPU_TYPE_DOUBLE; 
  else assert(false); 

  // dimensions 
  for (int i=0; i<ndims; i++) {
    fread(&dims[i], sizeof(int), 1, fp);
    if (datatype == GLGPU_TYPE_FLOAT) {
      float length; 
      fread(&length, sizeof(float), 1, fp);
      lengths[i] = length; 
    } else if (datatype == GLGPU_TYPE_DOUBLE) {
      fread(&lengths[i], sizeof(double), 1, fp); 
    }
  }

  // dummy
  int dummy; 
  fread(&dummy, sizeof(int), 1, fp);

  // time, fluctuation_amp, Bx, By, Bz, Jx
  if (datatype == GLGPU_TYPE_FLOAT) {
    float time_, fluctuation_amp_, B_[3], Jx_; 
    fread(&time_, sizeof(float), 1, fp);
    fread(&fluctuation_amp_, sizeof(float), 1, fp); 
    fread(&B_, sizeof(float), 3, fp);
    fread(&Jx_, sizeof(float), 1, fp); 
    time = time_; 
    // _fluctuation_amp = fluctuation_amp;
    B[0] = B_[0]; 
    B[1] = B_[1]; 
    B[2] = B_[2];
    Jxext = Jx_;
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    double fluctuation_amp;
    fread(&time, sizeof(double), 1, fp); 
    fread(&fluctuation_amp, sizeof(double), 1, fp);
    fread(B, sizeof(double), 3, fp);
    fread(&Jxext, sizeof(double), 1, fp); 
  }

  // btype
  int btype; 
  fread(&btype, sizeof(int), 1, fp); 
  pbc[0] = btype & 0x0000ff;
  pbc[1] = btype & 0x00ff00;
  pbc[2] = btype & 0xff0000; 

  // optype
  int optype; 
  fread(&optype, sizeof(int), 1, fp);
  if (datatype == GLGPU_TYPE_FLOAT) {
    float Kex_, Kex_dot_; 
    fread(&Kex_, sizeof(float), 1, fp);
    fread(&Kex_dot_, sizeof(float), 1, fp); 
    Kex = Kex_;
    // Kex_dot = Kex_dot_;
  } else if (datatype == GLGPU_TYPE_DOUBLE) {
    double Kex_dot;
    fread(&Kex, sizeof(double), 1, fp);
    fread(&Kex_dot, sizeof(double), 1, fp); 
  }

  int count = 1; 
  for (int i=0; i<ndims; i++) 
    count *= dims[i]; 

  int offset = ftell(fp);
  
  // mem allocation 
  *re = (double*)realloc(*re, sizeof(double)*count);
  *im = (double*)realloc(*im, sizeof(double)*count);

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
        (*re)[i] = ch1[i]; 
        (*im)[i] = ch2[i]; 
      }
    } else if (optype == 1) {
      for (int i=0; i<count; i++) {
        (*re)[i] = ch1[i] * cos(ch2[i]); 
        (*im)[i] = ch1[i] * sin(ch2[i]);
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
      _rho = (double*)malloc(sizeof(double)*count);
      _phi = (double*)malloc(sizeof(double)*count); 
      for (int i=0; i<count; i++) {
        _rho[i] = sqrt(_re[i]*_re[i] + _im[i]*_im[i]);
        _phi[i] = atan2(_im[i], _re[i]); 
      }
    } else if (optype == 1) {
      _rho = ch1; 
      _phi = ch2;
      _re = (double*)malloc(sizeof(double)*count); 
      _im = (double*)malloc(sizeof(double)*count); 
      for (int i=0; i<count; i++) {
        _re[i] = _rho[i] * cos(_phi[i]); 
        _im[i] = _rho[i] * sin(_phi[i]); 
      }
    } else assert(false);
#endif
  }
  return true;
}

#if 0
bool GLGPU_IO_Helper_WriteNetCDF(
    const std::string& filename, 
    int &ndims, 
    int *dims,
    double *lengths,
    bool *pbc,
    double *B,
    double &Jxext, 
    double &Kx, 
    double &V, 
    double **re, 
    double **im)
{
#ifdef WITH_LIBMESH
  int ncid; 
  int dimids[3]; 
  int varids[8];

  size_t starts[3] = {0, 0, 0}, 
         sizes[3]  = {(size_t)_dims[2], (size_t)_dims[1], (size_t)_dims[0]};

  const int cnt = sizes[0]*sizes[1]*sizes[2];
  double *rho = (double*)malloc(sizeof(double)*cnt), 
         *phi = (double*)malloc(sizeof(double)*cnt);
  for (int i=0; i<cnt; i++) {
    rho[i] = sqrt(_re[i]*_re[i] + _im[i]*_im[i]);
    phi[i] = atan2(_im[i], _re[i]);
  }

  fprintf(stderr, "filename=%s\n", filename.c_str());

  NC_SAFE_CALL( nc_create(filename.c_str(), NC_CLOBBER | NC_64BIT_OFFSET, &ncid) ); 
  NC_SAFE_CALL( nc_def_dim(ncid, "z", sizes[0], &dimids[0]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "y", sizes[1], &dimids[1]) );
  NC_SAFE_CALL( nc_def_dim(ncid, "x", sizes[2], &dimids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "rho", NC_DOUBLE, 3, dimids, &varids[0]) );
  NC_SAFE_CALL( nc_def_var(ncid, "phi", NC_DOUBLE, 3, dimids, &varids[1]) );
  NC_SAFE_CALL( nc_def_var(ncid, "re", NC_DOUBLE, 3, dimids, &varids[2]) );
  NC_SAFE_CALL( nc_def_var(ncid, "im", NC_DOUBLE, 3, dimids, &varids[3]) );
  NC_SAFE_CALL( nc_def_var(ncid, "Jx", NC_DOUBLE, 3, dimids, &varids[4]) );
  NC_SAFE_CALL( nc_def_var(ncid, "Jy", NC_DOUBLE, 3, dimids, &varids[5]) );
  NC_SAFE_CALL( nc_def_var(ncid, "Jz", NC_DOUBLE, 3, dimids, &varids[6]) );
  // NC_SAFE_CALL( nc_def_var(ncid, "scm", NC_DOUBLE, 3, dimids, &varids[7]) );
  NC_SAFE_CALL( nc_enddef(ncid) );

  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[0], starts, sizes, rho) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[1], starts, sizes, phi) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[2], starts, sizes, _re) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[3], starts, sizes, _im) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[4], starts, sizes, _Jx) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[5], starts, sizes, _Jy) ); 
  NC_SAFE_CALL( nc_put_vara_double(ncid, varids[6], starts, sizes, _Jz) ); 
  // NC_SAFE_CALL( nc_put_vara_double(ncid, varids[7], starts, sizes, _scm) ); 

  NC_SAFE_CALL( nc_close(ncid) );

  return true;
#else
  assert(false);
  return false;
#endif
}
#endif
