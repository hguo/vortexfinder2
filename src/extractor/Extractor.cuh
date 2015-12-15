#ifndef _EXTRACTOR_CUH
#define _EXTRACTOR_CUH

typedef struct {
  int fid; // chirality = fid>0 ? 1 : -1;
  float pos[3];
} gpu_pf_t; // punctured faces from GPU output, 16 bytes

void vfgpu_upload_data(
    const int d_[3], 
    const bool pbc_[3], 
    const float origins_[3],
    const float lengths_[3], 
    const float cell_lengths_[3],
    const float B_[3],
    float Kx_,
    const float *re, 
    const float *im);

void vfgpu_extract_faces_tet();

#endif
