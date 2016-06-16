#ifndef _VF2GPU_H
#define _VF2GPU_H

extern "C" {

enum {
  VF2GPU_MESH_HEX = 0, // 3D
  VF2GPU_MESH_TET,
  VF2GPU_MESH_QUAD, // 2D
  VF2GPU_MESH_TRI
};

enum {
  VF2GPU_GAUGE_YZ = 0,
  VF2GPU_GAUGE_XZ = 1
};

struct vf2gpu_ctx_t;
struct vf2gpu_hdr_t;
struct vf2gpu_pf_t;
struct vf2gpu_pe_t;

vf2gpu_ctx_t* vf2gpu_create_ctx();
void vf2gpu_destroy_ctx(vf2gpu_ctx_t*);

void vf2gpu_set_meshtype(vf2gpu_ctx_t*, int);
void vf2gpu_set_num_threads(vf2gpu_ctx_t*, int);
void vf2gpu_set_output(vf2gpu_ctx_t*, const char *filename);

void vf2gpu_set_data_gpu(
    vf2gpu_ctx_t*, 
    const vf2gpu_hdr_t &h, 
    float *d_re_im,
    float *d_tmp1, // requires exact length as d_re_im
    float *d_tmp2);

void vf2gpu_extract_vortices(vf2gpu_ctx_t*);

// void vf2gpu_extract_faces(vf2gpu_ctx_t*);
// void vf2gpu_get_pflist(vf2gpu_ctx_t*, int *n, vf2gpu_pf_t **pflist);
// void vf2gpu_trace_faces(vf2gpu_ctx_t*);

// NOTE: not efficient.  for debug purposes only
// int vf2gpu_write_binary(vf2gpu_ctx_t*, const char *filename);
// int vf2gpu_write_ascii(vf2gpu_ctx_t*, const char *filename);

struct vf2gpu_pf_t {
  unsigned int fid; 
  signed char chirality;
  float pos[3];
}; // punctured faces from GPU output, 16 bytes

struct vf2gpu_pe_t {
  unsigned int eid;
  signed char chirality;
};

struct vf2gpu_hdr_t {
  int d[3];
  unsigned int count; // d[0]*d[1]*d[2];
  bool pbc[3];
  float origins[3];
  float lengths[3];
  float cell_lengths[3];
  float B[3];
  float Kx;
};

struct vf2gpu_ctx_t {
  unsigned char meshtype; 
  
  vf2gpu_hdr_t h;
  vf2gpu_hdr_t *d_h;
  float *d_re_im, *d_rho_phi;

  unsigned int *d_pfcount;
  vf2gpu_pf_t *d_pflist;
  unsigned int pfcount;
  // vf2gpu_pf_t *pflist;

  // cuda related
  int blockSize, minGridSize, gridSize;
};

}

#endif
