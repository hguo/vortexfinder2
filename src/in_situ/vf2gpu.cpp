#include "vf2gpu.h"
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" {
///////////////////

vf2gpu_ctx_t* vf2gpu_create_ctx()
{
  vf2gpu_ctx_t *c = (vf2gpu_ctx_t*)malloc(sizeof(vf2gpu_ctx_t));
  memset(c, 0, sizeof(vf2gpu_ctx_t));

  // cudaOccupancyMaxPotentialBlockSize(
  //     &c->minGridSize, &c->blockSize, 
  //     extract_faces_kernel, 0, 0);

  return c;
}

void vf2gpu_destroy_ctx(vf2gpu_ctx_t *c)
{
  cudaFree(c->d_h);
  // free(c->pflist);
  free(c);
  // checkLastCudaError("[vf2gpu] destroying ctx");
}

void vf2gpu_set_meshtype(vf2gpu_ctx_t* c, int meshtype)
{
  c->meshtype = meshtype;
}

void vf2gpu_get_pflist(vf2gpu_ctx_t* c, int *n, vf2gpu_pf_t **pflist)
{
  *n = c->pfcount; 
  // *pflist = c->pflist;
}

///////////////////
} // extern "C"
