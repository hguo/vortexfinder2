#include "vf2gpu.h"
#include "utils.cuh"
#include "mesh.cuh"
#include "threadIdx.cuh"
#include "inverseInterpolation.cuh"
#include <cstdio>
#include <algorithm>

template <typename T> 
__device__
T line_integral(const vf2gpu_hdr_t& h, const T X0[], const T X1[], const T A0[], const T A1[]) 
{
  T dX[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  T A[3] = {A0[0] + A1[0], A0[1] + A1[1], A0[2] + A1[2]};

  for (int i=0; i<3; i++)
    if (dX[i] > h.lengths[i]/2) dX[i] -= h.lengths[i];
    else if (dX[i] < -h.lengths[i]/2) dX[i] += h.lengths[i];

  return 0.5 * inner_product(A, dX);
}

template <typename T, int gauge>
__device__
inline void magnetic_potential(const vf2gpu_hdr_t& h, T X[3], T A[3])
{
  // if (h.B[1]>0) { // yz gauge
  if (gauge == VF2GPU_GAUGE_YZ) {
    A[0] = -h.Kx; 
    A[1] = X[0] * h.B[2];
    A[2] = -X[0] * h.B[1];
  } else { // xz gauge
    A[0] = -X[1] * h.B[2] - h.Kx;
    A[1] = 0;
    A[2] = X[1] * h.B[0];
  }
}

template <typename T, int meshtype, int gauge>
__device__
inline bool get_face_values(
    const vf2gpu_hdr_t& h, 
    int fid, 
    T X[][3],
    T A[][3],
    T rho[],
    T phi[],
    const T *rho_phi_)
{
  const int nnodes = meshtype == VF2GPU_MESH_TET ? 3 : 4;
  int nidxs[nnodes][3], nids[nnodes];
  bool valid;
  
  if (meshtype == VF2GPU_MESH_TET) valid = fid2nodes3D_tet(h, fid, nidxs);
  else if (meshtype == VF2GPU_MESH_HEX) valid = fid2nodes3D_hex(h, fid, nidxs);
  else valid = fid2nodes2D(h, fid, nidxs);
  
  if (valid) {
    for (int i=0; i<nnodes; i++) {
      if (meshtype == VF2GPU_MESH_QUAD) {
        nids[i] = nidx2nid2D(h, nidxs[i]);
        nidx2pos2D(h, nidxs[i], X[i]);
      } else {
        nids[i] = nidx2nid3D(h, nidxs[i]);
        nidx2pos3D(h, nidxs[i], X[i]);
      }

      rho[i] = rho_phi_[nids[i]*2];
      phi[i] = rho_phi_[nids[i]*2+1];
      
      magnetic_potential<T, gauge>(h, X[i], A[i]); 
    }
  }

  return valid;
}

template <typename T>
__device__
inline int contour_chirality(
    const vf2gpu_hdr_t &h, 
    int nnodes, // nnodes <= 4
    const T phi[], 
    const T X[][3], 
    const T A[][3],
    T delta[])
{
  T phase_jump = 0;
  for (int i=0; i<nnodes; i++) {
    int j = (i+1) % nnodes;
    delta[i] = phi[j] - phi[i]; 
    T li = line_integral(h, X[i], X[j], A[i], A[j]), 
      qp = 0; // TODO
    delta[i] = mod2pi1(delta[i] - li + qp);
    phase_jump -= delta[i];
  }
  
  if (fabs(phase_jump)<0.5) return 0; // not punctured
  else return sgn(phase_jump);
}

// for space-time vfaces
template <typename T>
__device__
inline int contour_chirality_spt(
    const vf2gpu_hdr_t &h, 
    const vf2gpu_hdr_t &h1, 
    const T phi[4], 
    const T X[4][3], 
    const T A[4][3],
    T delta[])
{
  T li[4] = { // FIXME: varying B
    line_integral(h, X[0], X[1], A[0], A[1]), 
    0, 
    line_integral(h, X[1], X[0], A[2], A[3]), 
    0};
  T qp[4] = {0, 0, 0, 0}; // FIXME

  T phase_jump = 0;
  for (int i=0; i<4; i++) {
    int j = (i+1) % 4;
    delta[i] = phi[j] - phi[i]; 
    delta[i] = mod2pi1(delta[i] - li[i] + qp[i]);
    phase_jump -= delta[i];
  }
  
  if (fabs(phase_jump)<0.5) return 0; // not punctured
  else return sgn(phase_jump);
}

template <typename T>
__device__
inline void gauge_transform(
    int nnodes, 
    const T rho[],
    const T delta[],
    T phi[], 
    T re[], 
    T im[])
{
  re[0] = rho[0] * cos(phi[0]);
  im[0] = rho[0] * sin(phi[0]);
  for (int i=1; i<nnodes; i++) {
    phi[i] = phi[i-1] + delta[i-1];
    re[i] = rho[i] * cos(phi[i]);
    im[i] = rho[i] * sin(phi[i]);
  }
}

template <typename T, int meshtype, int gauge>
__device__
inline int extract_face(
    const vf2gpu_hdr_t& h, 
    int fid,
    unsigned int *pfcount,
    vf2gpu_pf_t *pflist,
    const T *rho_phi_)
{
  const int nnodes = meshtype == VF2GPU_MESH_TET ? 3 : 4;
  T X[nnodes][3], A[nnodes][3], rho[nnodes], phi[nnodes], re[nnodes], im[nnodes];
  T delta[nnodes];
  
  bool valid = get_face_values<T, meshtype, gauge>(h, fid, X, A, rho, phi, rho_phi_);
  if (!valid) return 0;

  // compute phase shift
  int chirality = contour_chirality(h, nnodes, phi, X, A, delta);
  if (chirality == 0) return 0;
  
  // gauge transformation
  gauge_transform(nnodes, rho, delta, phi, re, im);

  // find puncture point
  vf2gpu_pf_t pf; 
  pf.fid = fid;
  pf.chirality = chirality;
  find_zero<T, meshtype>(re, im, X, pf.pos, T(1));
  
  unsigned int idx = atomicInc(pfcount, 0xffffffff);
  pflist[idx] = pf;

  return chirality;
}

template <typename T>
__global__
static void compute_rho_phi_kernel(
    const vf2gpu_hdr_t *h,
    T *re_im, T *rho_phi)
{
  int idx = getGlobalIdx_3D_1D();
  if (idx>=h->count) return;

  T r, i;
  r = re_im[idx*2];
  i = re_im[idx*2+1];

  rho_phi[idx*2] = sqrt(r*r + i*i);
  rho_phi[idx*2+1] = atan2(i, r);
}

template <typename T>
__global__
static void copy_phi_kernel( // copy phi to another array
    const vf2gpu_hdr_t *h, 
    T *rho_phi, T *phi)
{
  int idx = getGlobalIdx_3D_1D();
  if (idx>=h->count) return;
  phi[idx] = rho_phi[idx*2+1];
}

template <typename T, int meshtype, int gauge>
__global__
static void extract_faces_kernel(
    const vf2gpu_hdr_t* h, 
    unsigned int *pfcount,
    const unsigned int pflimit,
    vf2gpu_pf_t *pflist,
    const T *rho_phi)
{
  int nfacetypes;
  if (meshtype == VF2GPU_MESH_TET) nfacetypes = 12;
  else if (meshtype == VF2GPU_MESH_HEX) nfacetypes = 3;
  else nfacetypes = 1;
  
  int fid = getGlobalIdx_3D_1D();
  if (fid>=h->count*nfacetypes) return;

#if 0 // use global memory
  extract_face<T, meshtype, gauge>(*h, fid, pfcount, pflist, rho, phi);
#else // use shared memory
  extern __shared__ char smem[];
  unsigned int *spfcount = (unsigned int*)smem;
  vf2gpu_pf_t *spflist= (vf2gpu_pf_t*)(smem + sizeof(int));
 
  if (threadIdx.x == 0)
    *spfcount = 0;
  __syncthreads();

  extract_face<T, meshtype, gauge>(*h, fid, spfcount, spflist, rho_phi);
  __syncthreads();

  if (threadIdx.x == 0 && (*spfcount)>0) {
    unsigned int idx = atomicAdd(pfcount, *spfcount);
    // printf("idx=%d, count=%d\n", idx, *spfcount);
    if (idx + *spfcount < pflimit)
      memcpy(pflist + idx, spflist, (*spfcount) * sizeof(vf2gpu_pf_t));
  }
#endif
}

////////////////////////////////
extern "C" {

void vf2gpu_set_data_insitu(
    vf2gpu_ctx_t* c,
    const vf2gpu_hdr_t &h,
    float *d_re_im,
    float *d_tmp1,
    float *d_tmp2)
{
  if (c->d_h == NULL) 
    cudaMalloc((void**)&c->d_h, sizeof(vf2gpu_hdr_t));
 
  memcpy(&c->h, &h, sizeof(vf2gpu_hdr_t));
  cudaMemcpy(c->d_h, &c->h, sizeof(vf2gpu_hdr_t), cudaMemcpyHostToDevice);

  c->d_re_im = d_re_im;
  c->d_rho_phi = d_tmp1;
  c->d_pfcount = (unsigned int*)d_tmp2;
  c->d_pflist = (vf2gpu_pf_t*)(d_tmp2 + 1);
}

void vf2gpu_compute_rho_phi(vf2gpu_ctx_t* c)
{
  const int count = c->h.count;
  const int maxGridDim = 1024; // 32768;
  const int blockSize = 256;
  const int nBlocks = idivup(count, blockSize);
  dim3 gridSize; 
  if (nBlocks >= maxGridDim) 
    gridSize = dim3(idivup(nBlocks, maxGridDim), maxGridDim);
  else 
    gridSize = dim3(nBlocks);

  // fprintf(stderr, "count=%d\n", c->h.count);
  compute_rho_phi_kernel<float><<<gridSize, blockSize>>>(c->d_h, c->d_re_im, c->d_rho_phi);

  checkLastCudaError("[vf2gpu] compute rho and phi");
}

void vf2gpu_copy_phi(vf2gpu_ctx_t* c)
{
  const int count = c->h.count;
  const int maxGridDim = 1024; // 32768;
  const int blockSize = 256;
  const int nBlocks = idivup(count, blockSize);
  dim3 gridSize; 
  if (nBlocks >= maxGridDim) 
    gridSize = dim3(idivup(nBlocks, maxGridDim), maxGridDim);
  else 
    gridSize = dim3(nBlocks);

  // copy_phi_kernel<float><<<gridSize, blockSize>>>(c->d_h, c->d_re_im, c->d_rho_phi);

  checkLastCudaError("[vf2gpu] copy phi");
}

void vf2gpu_extract_faces(vf2gpu_ctx_t* c)
{
  int nfacetypes;
  if (c->meshtype == VF2GPU_MESH_TET) nfacetypes = 12;
  else if (c->meshtype == VF2GPU_MESH_HEX) nfacetypes = 3;
  else nfacetypes = 1;
  
  const int threadCount = c->h.count * nfacetypes;
  const int maxGridDim = 1024; // 32768;
  const int blockSize = 256;
  const int nBlocks = idivup(threadCount, blockSize);
  dim3 gridSize; 
  if (nBlocks >= maxGridDim) 
    gridSize = dim3(idivup(nBlocks, maxGridDim), maxGridDim);
  else 
    gridSize = dim3(nBlocks);
  const int sharedSize = blockSize * sizeof(vf2gpu_pf_t) + sizeof(unsigned int); // enough shared memory for every block
  const unsigned int pflimit = c->h.count / sizeof(vf2gpu_pf_t);
  const int gauge = c->h.B[1]>0 ? VF2GPU_GAUGE_YZ : VF2GPU_GAUGE_XZ;
 
  vf2gpu_compute_rho_phi(c);

  cudaMemset(c->d_pfcount, 0, sizeof(unsigned int));

  if (c->meshtype == VF2GPU_MESH_HEX) {
    if (gauge == VF2GPU_GAUGE_YZ) 
      extract_faces_kernel<float, VF2GPU_MESH_HEX, VF2GPU_GAUGE_YZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
    else
      extract_faces_kernel<float, VF2GPU_MESH_HEX, VF2GPU_GAUGE_XZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
  } else if (c->meshtype == VF2GPU_MESH_TET) {
    if (gauge == VF2GPU_GAUGE_YZ) 
      extract_faces_kernel<float, VF2GPU_MESH_TET, VF2GPU_GAUGE_YZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
    else
      extract_faces_kernel<float, VF2GPU_MESH_TET, VF2GPU_GAUGE_XZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
  } else if (c->meshtype == VF2GPU_MESH_QUAD) {
    if (gauge == VF2GPU_GAUGE_YZ) 
      extract_faces_kernel<float, VF2GPU_MESH_QUAD, VF2GPU_GAUGE_YZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
    else
      extract_faces_kernel<float, VF2GPU_MESH_QUAD, VF2GPU_GAUGE_XZ><<<gridSize, blockSize, sharedSize>>>
        (c->d_h, c->d_pfcount, pflimit, c->d_pflist, c->d_rho_phi);
  }

  cudaMemcpy((void*)&c->pfcount, c->d_pfcount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("pfcount=%d\n", c->pfcount);
  // if (c->pfcount>0)
  //   cudaMemcpy(c->pflist, c->d_pflist, sizeof(vf2gpu_pf_t)*c->pfcount, cudaMemcpyDeviceToHost);
  
  checkLastCudaError("[vf2gpu] extract faces");
}

int vf2gpu_write_binary(vf2gpu_ctx_t* c, const char *filename)
{
  int pfcount;
  cudaMemcpy(&pfcount, c->d_pfcount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  const int pflimit = c->h.count / sizeof(vf2gpu_pf_t);
  if (pfcount == 0 || pfcount>pflimit) return 0;

  vf2gpu_pf_t *pflist = (vf2gpu_pf_t*)malloc(sizeof(vf2gpu_pf_t) * pfcount);
  cudaMemcpy(pflist, c->d_pflist, sizeof(vf2gpu_pf_t)*pfcount, cudaMemcpyDeviceToHost);

  FILE *fp = fopen(filename, "wb");
  fwrite(&pfcount, sizeof(int), 1, fp);
  fwrite(pflist, sizeof(vf2gpu_pf_t), pfcount, fp);
  fclose(fp);

  free(pflist);
  return pfcount;
}

int vf2gpu_write_ascii(vf2gpu_ctx_t* c, const char *filename)
{
  int pfcount;
  cudaMemcpy(&pfcount, c->d_pfcount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  const int pflimit = c->h.count / sizeof(vf2gpu_pf_t);
  if (pfcount == 0 || pfcount>pflimit) return 0;

  vf2gpu_pf_t *pflist = (vf2gpu_pf_t*)malloc(sizeof(vf2gpu_pf_t) * pfcount);
  cudaMemcpy(pflist, c->d_pflist, sizeof(vf2gpu_pf_t)*pfcount, cudaMemcpyDeviceToHost);

  FILE *fp = fopen(filename, "w");
  fprintf(fp, "%d\n", pfcount);
  for (int i=0; i<pfcount; i++) 
    fprintf(fp, "%d, %d, %f, %f, %f\n", pflist[i].fid, pflist[i].chirality, pflist[i].pos[0], pflist[i].pos[1], pflist[i].pos[2]);
  fclose(fp);

  free(pflist);
  return pfcount;
}

} // extern "C"
