#ifndef _VFGPU_H
#define _VFGPU_H

extern "C" {

enum {
  VFGPU_MESH_HEX = 0, // 3D
  VFGPU_MESH_TET,
  VFGPU_MESH_QUAD, // 2D
  VFGPU_MESH_TRI
};

enum {
  VFGPU_GAUGE_YZ = 0,
  VFGPU_GAUGE_XZ = 1
};

struct vfgpu_ctx_t;
struct vfgpu_hdr_t;
struct vfgpu_pf_t;
struct vfgpu_pe_t;

vfgpu_ctx_t* vfgpu_create_ctx();
void vfgpu_destroy_ctx(vfgpu_ctx_t*);

void vfgpu_set_meshtype(vfgpu_ctx_t*, int);
void vfgpu_set_num_threads(vfgpu_ctx_t*, int);
void vfgpu_set_output(vfgpu_ctx_t*, const char *filename);

void vfgpu_set_data_gpu(
    vfgpu_ctx_t*, 
    const vfgpu_hdr_t &h, 
    float *d_re_im,
    float *d_tmp1, // requires exact length as d_re_im
    float *d_tmp2);

void vfgpu_extract_vortices(vfgpu_ctx_t*);

// void vfgpu_extract_faces(vfgpu_ctx_t*);
// void vfgpu_get_pflist(vfgpu_ctx_t*, int *n, vfgpu_pf_t **pflist);
// void vfgpu_trace_faces(vfgpu_ctx_t*);

// NOTE: not efficient.  for debug purposes only
// int vfgpu_write_binary(vfgpu_ctx_t*, const char *filename);
// int vfgpu_write_ascii(vfgpu_ctx_t*, const char *filename);
}

#endif
