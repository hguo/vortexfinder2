#ifndef _RCCUDA_H
#define _RCCUDA_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
  RCKERNEL_UCHAR = 0, 
  RCKERNEL_USHORT= 1,
  RCKERNEL_FLOAT = 2, 
  RCKERNEL_DOUBLE = 3
}; 

enum {
  RCTRANSFORM_DISABLED = 0, 
  RCTRANSFORM_ENABLED  = 1 
}; 

enum {
  RCSHADING_NONE  = 0, 
  RCSHADING_PHONG = 1, 
  RCSHADING_COOK  = 2
};

enum {
  RCBLOCKING_DISABLED = 0,
  RCBLOCKING_ENABLED = 1
}; 

struct cudaArray; 

struct ctx_rc {
  int viewport[4];
  float projmatrix[16], mvmatrix[16], invmvp[16]; 

  float *d_output;
  cudaArray *d_volume; 
  cudaArray *d_tf; 
  int dsz[3];  // domain size
  int st[3], sz[3], gst[3], gsz[3]; 
  int size_tf;
  float stepsize;
  float trans[2];

  int blocking; 
  int transform; 
  int shading;
  int rckernel;
};  

void rc_create_ctx(ctx_rc **ctx); 
void rc_destroy_ctx(ctx_rc **ctx); 

void rc_bind_transfer_function_array(cudaArray* array); 

void rc_bind_volume_uchar(ctx_rc *ctx, float *buf, int gsz[3]); 
void rc_bind_volume_ushort(ctx_rc *ctx, float *buf, int gsz[3]); 
void rc_bind_volume_float(ctx_rc *ctx, const float *buf, const int st[3], const int sz[3], const int gst[3], const int gsz[3]); 

void rc_set_kernel(ctx_rc *ctx, int kernel); 
void rc_set_dsz(ctx_rc *ctx, int xsz, int ysz, int zsz);
void rc_set_stepsize(ctx_rc *ctx, float stepsize); 
void rc_set_viewport(ctx_rc *ctx, int x, int y, int w, int h);
void rc_set_invmvpf(ctx_rc *ctx, float *invmvp); 
void rc_set_invmvpd(ctx_rc *ctx, double *invmvp); 
void rc_set_range(ctx_rc *ctx, float a, float b); 
void rc_render(ctx_rc *ctx);

void rc_clear_output(ctx_rc *ctx); 
void rc_dump_output(ctx_rc *ctx, float *output); 

#ifdef __cplusplus
}
#endif

#endif
