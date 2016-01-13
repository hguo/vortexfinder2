#ifndef _EXTRACTOR_CUH
#define _EXTRACTOR_CUH

#include "def.h"

typedef struct {
  int fid; 
  int chirality;
  float pos[3];
} gpu_pf_t; // punctured faces from GPU output, 16 bytes

typedef struct {
  int eid;
  int chirality;
} gpu_pe_t;

void vfgpu_upload_data(
    int slot,
    const int d_[3], 
    const bool pbc_[3], 
    const float origins_[3],
    const float lengths_[3], 
    const float cell_lengths_[3],
    const float B_[3],
    float Kx_,
    const float *re, 
    const float *im);

void vfgpu_destroy_data();

void vfgpu_extract_faces(int slot, int *pfcount, gpu_pf_t **pfbuf, int discretization=GLGPU3D_MESH_HEX);

#endif
