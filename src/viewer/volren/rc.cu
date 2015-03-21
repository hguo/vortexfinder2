/*
PKUVIS CONFIDENTIAL
___________________

Copyright (c) 2009-2012, PKU Visualization and Visual Analytics Group 
Produced at Peking University, Beijing, China.
All rights reserved.
                                                                             
NOTICE: THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF  VISUALIZATION 
AND VISUAL ANALYTICS GROUP (PKUVIS), PEKING UNIVERSITY. DISSEMINATION
OF  THIS  INFORMATION  OR  REPRODUCTION OF THIS  MATERIAL IS STRICTLY 
FORBIDDEN UNLESS PRIOR WRITTEN PERMISSION IS OBTAINED FROM PKUVIS.
*/

#include "rccommon.cuh"
#include "rc.h"

texture<unsigned char, 3, cudaReadModeNormalizedFloat> texVolumeUchar;
texture<unsigned short, 3, cudaReadModeNormalizedFloat> texVolumeUshort; 
texture<float, 3, cudaReadModeElementType> texVolumeFloat; 
texture<double, 3, cudaReadModeElementType> texVolumeDouble; 
texture<float4, 1, cudaReadModeElementType> texTransferFunc;

__constant__ int c_viewport[4];
__constant__ float c_invmvp[16]; 

template <class DataType, enum cudaTextureReadMode readMode, int TRANSFORM>
__device__ static inline float tex3Dtrans(
        texture<DataType, cudaTextureType3D, readMode> texRef, 
        float2 trans, float3 coords)
{
  if (TRANSFORM)
    return trans.x * tex3D(texRef, coords.x, coords.y, coords.z) + trans.y; 
  else 
    return tex3D(texRef, coords.x, coords.y, coords.z); 
}

template <class DataType, enum cudaTextureReadMode readMode, int BLOCKING, int TRANSFORM, int SHADING>
__device__ static void raycasting(
        float4 &dst,              // destination color 
        texture<DataType, cudaTextureType3D, readMode> texVolume, 
                                  // volume texture 
        float2 trans,             // range transformation 
        float3 rayO,              // ray origin 
        float3 rayD,              // ray direction
        float stepsize,           // stepsize
        float3 dsz,               // domain size
        float3 st,                // block start
        float3 sz,                // block size
        float3 gst,               // block ghost start
        float3 gsz)               // block ghost size
{
  float4 src = make_float4(0); 
  float3 pos; // actual position 
  float3 coords; // unnormalized tex coordinates 
  float3 ratio = make_float3(1.f/gsz.x, 1.f/gsz.y, 1.f/gsz.z); // for blocking 
  float  sample = 0.f; 
  float3 N, L = make_float3(1, 0, 0), V = rayD; 
  float3 Ka = make_float3(0.04), 
         Kd = make_float3(0.3), 
         Ks = make_float3(0.2); 
  const float delta = 0.5f / dsz.x;   // for shading 

  float tnear0, tfar0, // global
        tnear, tfar;   // local
  
  if (BLOCKING) {
    if (!intersectBox(rayO, rayD, tnear0, tfar0, make_float3(0.f), dsz)) return; // intersect with domain 
    if (!intersectBox(rayO, rayD, tnear, tfar, sz, sz+st)) return; // intersect with block 
    tnear = ceilf((tnear - tnear0) / stepsize) * stepsize + tnear0; // align the ray with the global entry point
  } else {
    if (!intersectBox(rayO, rayD, tnear, tfar, make_float3(0.f), dsz)) return; 
  }
  float t = max(0.f, tnear); // looking inside the volume

  while(1) { // raycasting
    pos = rayO + rayD*t;
   
    if (BLOCKING) coords = (pos - gst) * ratio;
    else coords = pos;

    sample = tex3Dtrans<DataType, readMode, TRANSFORM>(texVolume, trans, coords); 
    // src = tex1D(texTransferFunc, sample);
    // src = make_float4(sample, 1.0-sample, 0.0, 0.9);
    sample = pow(1.f - sample, 2.f); 
    src = make_float4(sample*2, 1.f-sample*2, 0.0, sample*0.4); 
    
    if (SHADING) {
      float3 lit; 
      N = gradient(texVolume, coords, delta); 
      lit = cook(N, V, L, Ka, Kd, Ks); 
      src.x += lit.x; 
      src.y += lit.y; 
      src.z += lit.z; 
    }

    src.w = 1.f - pow(1.f - src.w, stepsize); // alpha correction  

    dst.x += (1.0 - dst.w) * src.x * src.w;
    dst.y += (1.0 - dst.w) * src.y * src.w;
    dst.z += (1.0 - dst.w) * src.z * src.w;
    dst.w += (1.0 - dst.w) * src.w;
    
    t += stepsize; 
    
    if (t>tfar) break; // no early ray termination in compositing mode
    // if (t>tfar || dst.w>opacityThreshold) break;
  }
}

template <int KERNEL, int BLOCKING, int TRANSFORM, int SHADING>
__global__ static void raycasting_kernel(
        float *output, 
        float2 trans, 
        float stepsize, 
        float3 dsz, 
        float3 st, 
        float3 sz, 
        float3 gst, 
        float3 gsz)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= c_viewport[2] || y>= c_viewport[3]) return;
  
  float coord[4], obj0[4], obj1[4]; 
  coord[0] = (x-c_viewport[0])*2.f / c_viewport[2] - 1.f; 
  coord[1] = (y-c_viewport[1])*2.f / c_viewport[3] - 1.f; 
  coord[2] = -1.0; 
  coord[3] = 1.0;

  mulmatvec(c_invmvp, coord, obj0); 
  coord[2] = 1.0; 
  mulmatvec(c_invmvp, coord, obj1); 
  if (obj0[3] == 0.f || obj1[3] == 0.f) return; 

  for (int i=0; i<3; i++)
      obj0[i] /= obj0[3], obj1[i] /= obj1[3]; 

  float3 rayO = make_float3(obj0[0], obj0[1], obj0[2]), 
         rayD = normalize(make_float3(obj1[0]-obj0[0], obj1[1]-obj0[1], obj1[2]-obj0[2]));

  float4 dst = make_float4(0.f); 

  switch (KERNEL) {
  case RCKERNEL_UCHAR:
      raycasting<unsigned char, cudaReadModeNormalizedFloat, BLOCKING, TRANSFORM, SHADING>(
                  dst, 
                  texVolumeUchar, trans, 
                  rayO, rayD, stepsize,  
                  dsz, st, sz, gst, gsz); 
      break; 

  case RCKERNEL_USHORT:
      raycasting<unsigned short, cudaReadModeNormalizedFloat, BLOCKING, TRANSFORM, SHADING>(
                  dst, 
                  texVolumeUshort, trans, 
                  rayO, rayD, stepsize,  
                  dsz, st, sz, gst, gsz); 
      break; 
  
  case RCKERNEL_FLOAT: 
      raycasting<float, cudaReadModeElementType, BLOCKING, TRANSFORM, SHADING>(
                  dst, 
                  texVolumeFloat, trans, 
                  rayO, rayD, stepsize,  
                  dsz, st, sz, gst, gsz); 
      break; 

  default: break;
  }

  // GL_ONE_MINUS_DST_ALPHA, GL_ONE
  float w0 = 1-output[(y*c_viewport[2]+x)*4+3]; //, w1 = 1; make the compiler happy :)

  output[(y*c_viewport[2]+x)*4+0] += w0* dst.x;
  output[(y*c_viewport[2]+x)*4+1] += w0* dst.y;
  output[(y*c_viewport[2]+x)*4+2] += w0* dst.z;
  output[(y*c_viewport[2]+x)*4+3] += w0* dst.w;
}


/////////////////////////////
extern "C" {

void rc_render(ctx_rc *ctx)
{
  const dim3 blockSize(16, 16); 
  const dim3 gridSize = dim3(iDivUp(ctx->viewport[2], blockSize.x), iDivUp(ctx->viewport[3], blockSize.y));

  cudaMemcpyToSymbol(c_viewport, ctx->viewport, sizeof(int)*4);
  cudaMemcpyToSymbol(c_invmvp, ctx->invmvp, sizeof(float)*16);

  switch (ctx->rckernel) {
  case RCKERNEL_UCHAR: 
      raycasting_kernel<RCKERNEL_UCHAR, RCBLOCKING_DISABLED, RCTRANSFORM_DISABLED, RCSHADING_NONE><<<gridSize, blockSize>>>(
              ctx->d_output, 
              make_float2(ctx->trans[0], ctx->trans[1]), ctx->stepsize,  
              make_float3(ctx->dsz[0], ctx->dsz[1], ctx->dsz[2]), 
              make_float3(ctx->st[0], ctx->st[1], ctx->st[2]), 
              make_float3(ctx->sz[0], ctx->sz[1], ctx->sz[2]), 
              make_float3(ctx->gst[0], ctx->gst[1], ctx->gst[2]), 
              make_float3(ctx->gsz[0], ctx->gsz[1], ctx->gsz[2])); 
      break; 

  case RCKERNEL_USHORT:
      raycasting_kernel<RCKERNEL_USHORT, RCBLOCKING_DISABLED, RCTRANSFORM_ENABLED, RCSHADING_NONE><<<gridSize, blockSize>>>(
              ctx->d_output, 
              make_float2(ctx->trans[0], ctx->trans[1]), ctx->stepsize,  
              make_float3(ctx->dsz[0], ctx->dsz[1], ctx->dsz[2]), 
              make_float3(ctx->st[0], ctx->st[1], ctx->st[2]), 
              make_float3(ctx->sz[0], ctx->sz[1], ctx->sz[2]), 
              make_float3(ctx->gst[0], ctx->gst[1], ctx->gst[2]), 
              make_float3(ctx->gsz[0], ctx->gsz[1], ctx->gsz[2])); 
      break; 

  case RCKERNEL_FLOAT:
      raycasting_kernel<RCKERNEL_FLOAT, RCBLOCKING_DISABLED, RCTRANSFORM_ENABLED, RCSHADING_COOK><<<gridSize, blockSize>>>(
              ctx->d_output, 
              make_float2(ctx->trans[0], ctx->trans[1]), ctx->stepsize,  
              make_float3(ctx->dsz[0], ctx->dsz[1], ctx->dsz[2]), 
              make_float3(ctx->st[0], ctx->st[1], ctx->st[2]), 
              make_float3(ctx->sz[0], ctx->sz[1], ctx->sz[2]), 
              make_float3(ctx->gst[0], ctx->gst[1], ctx->gst[2]), 
              make_float3(ctx->gsz[0], ctx->gsz[1], ctx->gsz[2])); 
      break; 

  default: break; 
  }

  checkLastCudaError("[rc_render]");
}

#if 0
void rc_bind_volume_uchar_array(cudaArray *array)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

  texVolumeUchar.normalized = true; 
  texVolumeUchar.filterMode = cudaFilterModeLinear; 
  texVolumeUchar.addressMode[0] = cudaAddressModeClamp; 
  texVolumeUchar.addressMode[1] = cudaAddressModeClamp; 
  texVolumeUchar.addressMode[2] = cudaAddressModeClamp; 
  cudaBindTextureToArray(texVolumeUchar, array, channelDesc); 
  
  checkLastCudaError("[bind_bind_volume_uchar_array]");
}

void rc_bind_volume_ushort_array(cudaArray *array)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned short>();

  texVolumeUshort.normalized = true; 
  texVolumeUshort.filterMode = cudaFilterModeLinear; 
  texVolumeUshort.addressMode[0] = cudaAddressModeClamp; 
  texVolumeUshort.addressMode[1] = cudaAddressModeClamp; 
  texVolumeUshort.addressMode[2] = cudaAddressModeClamp; 
  cudaBindTextureToArray(texVolumeUshort, array, channelDesc); 
  
  checkLastCudaError("[rc_bind_volume_ushort_array]");
}
#endif

void rc_bind_transfer_function_array(cudaArray* array)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); 

  texTransferFunc.normalized = true; 
  texTransferFunc.filterMode = cudaFilterModeLinear; 
  texTransferFunc.addressMode[0] = cudaAddressModeClamp; 
  cudaBindTextureToArray(texTransferFunc, array, channelDesc); 

  checkLastCudaError("[rc_bind_transfer_function_array]");
}

void rc_bind_volume_float(ctx_rc *ctx, const float *buf, const int st[3], const int sz[3], const int gst[3], const int gsz[3])
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(gsz[0], gsz[1], gsz[2]); 
  void *ptr = (void*)buf;

#if 0
  if (!(ctx->gsz[0] == gsz[0] && ctx->gsz[1] == gsz[1] && ctx->gsz[2] == gsz[2])) {
    if (ctx->d_volume != NULL) 
      cudaFreeArray(ctx->d_volume);

    cudaMalloc3DArray(&ctx->d_volume, &channelDesc, extent); 
  }
#endif
  cudaMalloc3DArray(&ctx->d_volume, &channelDesc, extent); 
  
  memcpy(ctx->st, st, sizeof(int)*3); 
  memcpy(ctx->sz, sz, sizeof(int)*3); 
  memcpy(ctx->gst, gst, sizeof(int)*3); 
  memcpy(ctx->gsz, gsz, sizeof(int)*3); 
  
  cudaMemcpy3DParms copyParms = {0};
  copyParms.srcPtr   = make_cudaPitchedPtr(ptr, extent.width*sizeof(float), extent.width, extent.height); 
  copyParms.dstArray = ctx->d_volume; 
  copyParms.extent   = extent; 
  copyParms.kind     = cudaMemcpyHostToDevice; 
  cudaMemcpy3D(&copyParms); 
  
  texVolumeFloat.normalized = false;  
  texVolumeFloat.filterMode = cudaFilterModeLinear; 
  texVolumeFloat.addressMode[0] = cudaAddressModeClamp; 
  texVolumeFloat.addressMode[1] = cudaAddressModeClamp; 
  texVolumeFloat.addressMode[2] = cudaAddressModeClamp; 
  cudaBindTextureToArray(texVolumeFloat, ctx->d_volume, channelDesc); 
  
  checkLastCudaError("[rc_bind_volume_float]");
}

void rc_create_ctx(ctx_rc **ctx)
{
  *ctx = (ctx_rc*)malloc(sizeof(ctx_rc));
  memset(*ctx, 0, sizeof(ctx_rc));

  cudaMalloc((void**)&((*ctx)->d_output), sizeof(float)*4096*4096); 
}

void rc_destroy_ctx(ctx_rc **ctx)
{
  // TODO: free any resources

  free(*ctx); 
  *ctx = NULL; 
}

void rc_set_kernel(ctx_rc *ctx, int kernel)
{
  ctx->rckernel = kernel; 
}

void rc_set_viewport(ctx_rc *ctx, int x, int y, int w, int h)
{
  ctx->viewport[0] = x; 
  ctx->viewport[1] = y; 
  ctx->viewport[2] = w; 
  ctx->viewport[3] = h; 
}

void rc_set_range(ctx_rc *ctx, float a, float b)
{
  float c = 1.f/(b-a);
  ctx->trans[0] = c; 
  ctx->trans[1] = -a*c; 
}

void rc_set_stepsize(ctx_rc *ctx, float stepsize)
{
  ctx->stepsize = stepsize;
}

void rc_set_dsz(ctx_rc *ctx, int x, int y, int z)
{
  ctx->dsz[0] = x; 
  ctx->dsz[1] = y; 
  ctx->dsz[2] = z; 
}

void rc_set_invmvpf(ctx_rc *ctx, float *invmvp)
{
  memcpy(ctx->invmvp, invmvp, sizeof(float)*16); 
}

void rc_set_invmvpd(ctx_rc *ctx, double *invmvp)
{
  for (int i=0; i<16; i++) {
    ctx->invmvp[i] = invmvp[i]; 
  }
}

void rc_clear_output(ctx_rc *ctx)
{
  cudaMemset(ctx->d_output, 0, 4*sizeof(float)*ctx->viewport[2]*ctx->viewport[3]);
}

void rc_dump_output(ctx_rc *ctx, float *output)
{
  cudaMemcpy(output, ctx->d_output, 4*sizeof(float)*ctx->viewport[2]*ctx->viewport[3], cudaMemcpyDeviceToHost); 
}

} // extern "C" 
/////////////////
