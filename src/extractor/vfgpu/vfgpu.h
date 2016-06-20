#ifndef _EXTRACTOR_CUH
#define _EXTRACTOR_CUH

enum {
  VFGPU_MESH_HEX,
  VFGPU_MESH_TET
};

typedef struct {
  int fid; 
  int chirality;
  float pos[3];
} vfgpu_pf_t; // punctured faces from GPU output, 16 bytes

typedef struct {
  int eid;
  int chirality;
} vfgpu_pe_t;

typedef struct {
  int d[3];
  int count; // d[0]*d[1]*d[2];
  bool pbc[3];
  float origins[3];
  float lengths[3];
  float cell_lengths[3];
  float B[3];
  float Kx;
} vfgpu_hdr_t;

struct vfgpu_ctx_t;

vfgpu_ctx_t* vfgpu_create_ctx();
vfgpu_ctx_t* vfgpu_create_ctx_in_situ();
void vfgpu_destroy_ctx(vfgpu_ctx_t*);

void vfgpu_set_meshtype(vfgpu_ctx_t*, int);
void vfgpu_set_enable_count_lines_in_cell(vfgpu_ctx_t*, bool);
void vfgpu_set_pertubation(vfgpu_ctx_t*, float);

void vfgpu_upload_data(
    vfgpu_ctx_t*, 
    int slot,
    const vfgpu_hdr_t &h, 
    const float *re, 
    const float *im);

void vfgpu_set_data(
    vfgpu_ctx_t*, 
    int slot, 
    const vfgpu_hdr_t &h, 
    const float *psi_re_im);

void vfgpu_rotate_timesteps(vfgpu_ctx_t*);

void vfgpu_extract_faces(vfgpu_ctx_t*, int slot);

void vfgpu_extract_edges(vfgpu_ctx_t*);

void vfgpu_get_pflist(vfgpu_ctx_t*, int *n, vfgpu_pf_t **pflist);

void vfgpu_get_pelist(vfgpu_ctx_t*, int *n, vfgpu_pe_t **pelist);

void vfgpu_clear_count_lines_in_cell(vfgpu_ctx_t* c);

void vfgpu_count_lines_in_cell(vfgpu_ctx_t* c, int slot);

void vfgpu_dump_count_lines_in_cell(vfgpu_ctx_t* c);

#endif
