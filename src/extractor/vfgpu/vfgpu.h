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

typedef struct {
  int d[3];
  int count; // d[0]*d[1]*d[2];
  bool pbc[3];
  float origins[3];
  float lengths[3];
  float cell_lengths[3];
  float B[3];
  float Kx;
} gpu_hdr_t;

struct ctx_vfgpu_t;

ctx_vfgpu_t* vfgpu_create_ctx();
void vfgpu_destroy_ctx(ctx_vfgpu_t*);

void vfgpu_set_meshtype(ctx_vfgpu_t*, int);
void vfgpu_set_enable_count_lines_in_cell(ctx_vfgpu_t*, bool);
void vfgpu_set_pertubation(ctx_vfgpu_t*, float);

void vfgpu_upload_data(
    ctx_vfgpu_t*, 
    int slot,
    const gpu_hdr_t &h, 
    const float *re, 
    const float *im);

void vfgpu_rotate_timesteps(ctx_vfgpu_t*);

void vfgpu_extract_faces(ctx_vfgpu_t*, int slot);

void vfgpu_extract_edges(ctx_vfgpu_t*);

void vfgpu_get_pflist(ctx_vfgpu_t*, int *n, gpu_pf_t **pflist);

void vfgpu_get_pelist(ctx_vfgpu_t*, int *n, gpu_pe_t **pelist);

void vfgpu_clear_count_lines_in_cell(ctx_vfgpu_t* c);

void vfgpu_count_lines_in_cell(ctx_vfgpu_t* c, int slot);

void vfgpu_dump_count_lines_in_cell(ctx_vfgpu_t* c);

#endif
