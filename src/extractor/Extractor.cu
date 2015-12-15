#include "Extractor.cuh"
#include <cstdio>

__constant__ int d[3];
__constant__ bool pbc[3];
__constant__ float origins[3];
__constant__ float lengths[3];
__constant__ float cell_lengths[3];
__constant__ float B[3]; // TODO: time-varying data
__constant__ float Kx;

static int dims[3];
static float *d_rho=NULL, *d_phi=NULL, *d_re=NULL, *d_im=NULL;

static inline int idivup(int a, int b)
{
  return (a%b!=0) ? (a/b+1) : (a/b); 
}

template <typename T>
__device__
inline static T fmod1(T x, T y)
{
  T z = fmod(x, y);
  if (z<0) z += y;
  return z;
}

template <typename T>
__device__
inline static T mod2pi(T x)
{
  T y = fmod(x, 2*M_PI); 
  if (y<0) y+= 2*M_PI;
  return y; 
}

template <typename T>
__device__
inline static T mod2pi1(T x)
{
  return mod2pi(x + M_PI) - M_PI;
}

template <typename T> 
__device__
inline int sgn(T x) 
{
  return (T(0) < x) - (x < T(0));
}

template <typename T>
__device__
static inline T inner_product(const T A[3], const T B[3])
{
  return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

template <typename T> 
__device__
T line_integral(const T X0[], const T X1[], const T A0[], const T A1[]) 
{
  T dX[3] = {X1[0] - X0[0], X1[1] - X0[1], X1[2] - X0[2]};
  T A[3] = {A0[0] + A1[0], A0[1] + A1[1], A0[2] + A1[2]};

  for (int i=0; i<3; i++)
    if (dX[i] > lengths[i]/2) dX[i] -= lengths[i];
    else if (dX[i] < -lengths[i]/2) dX[i] += lengths[i];

  return 0.5 * inner_product(A, dX);
}

__device__
inline void nid2nidx(int id, int idx[3])
{
  int s = d[0] * d[1]; 
  int k = id / s; 
  int j = (id - k*s) / d[0]; 
  int i = id - k*s - j*d[0]; 

  idx[0] = i; idx[1] = j; idx[2] = k;
}

__device__
inline int nidx2nid(const int idx_[3])
{
  int idx[3] = {idx_[0], idx_[1], idx_[2]};
  for (int i=0; i<3; i++) {
    idx[i] = idx[i] % d[i];
    if (idx[i] < 0)
      idx[i] += d[i];
  }
  return idx[0] + d[0] * (idx[1] + d[1] * idx[2]); 
}

__device__
inline int fidx2fid_tet(const int idx[4])
{
  return nidx2nid(idx)*12 + idx[3];
}

__device__
inline void fid2fidx_tet(int id, int idx[4])
{
  int nid = id / 12;
  nid2nidx(nid, idx);
  idx[3] = id % 12;
}

__device__
bool valid_fidx_tet(const int fidx[4])
{
  if (fidx[3]<0 || fidx[3]>=12) return false;
  else {
    int o[3] = {0};
    for (int i=0; i<3; i++)
      if (pbc[i]) {
        if (fidx[i] < 0 || fidx[i] >= d[i]) return false;
      } else {
        if (fidx[i] < 0 || fidx[i] > d[i]-1) return false;
        else if (fidx[i] == d[i]-1) o[i] = 1;
      }
    
    const int sum = o[0] + o[1] + o[2];
    if (sum == 0) return true;
    else if (o[0] + o[1] + o[2] > 1) return false;
    else if (o[0] && (fidx[3] == 4 || fidx[3] == 5)) return true;
    else if (o[1] && (fidx[3] == 2 || fidx[3] == 3)) return true; 
    else if (o[2] && (fidx[3] == 0 || fidx[3] == 1)) return true;
    else return false;
  }
}

__device__
inline int eidx2eid_tet(const int idx[4])
{
  return nidx2nid(idx)*7 + idx[3];
}

__device__
inline void eid2eidx_tet(int id, int idx[4])
{
  int nid = id / 7;
  nid2nidx(nid, idx);
  idx[3] = id % 7;
}

__device__
inline
bool valid_eidx(const int eidx[4])
{
  if (eidx[3]<0 || eidx[3]>=7) return false;
  else {
    for (int i=0; i<3; i++)
      if (pbc[i]) {
        if (eidx[i] < 0 || eidx[i] >= d[i]) return false;
      } else {
        if (eidx[i] < 0 || eidx[i] >= d[i]-1) return false;
      }
    return true;
  }
}

__device__
inline bool fid2nodes_tet(int id, int nidxs[3][3])
{
  const int nodes_idx[12][3][3] = { // 12 types of faces
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}, // ABC
    {{0, 0, 0}, {1, 1, 0}, {0, 1, 0}}, // ACD
    {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}}, // ABF
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}}, // AEF
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}, // ADE
    {{0, 1, 0}, {0, 0, 1}, {0, 1, 1}}, // DEH
    {{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}, // ADF
    {{0, 1, 0}, {1, 0, 1}, {1, 1, 1}}, // DFG
    {{0, 1, 0}, {0, 0, 1}, {1, 0, 1}}, // DEF
    {{1, 1, 0}, {0, 1, 0}, {1, 0, 1}}, // CDF
    {{0, 0, 0}, {1, 1, 0}, {1, 0, 1}}, // ACF
    {{0, 1, 0}, {0, 0, 1}, {1, 1, 1}}  // DEG
  };

  int fidx[4];
  fid2fidx_tet(id, fidx);

  if (valid_fidx_tet(fidx)) {
    const int type = fidx[3];
    for (int p=0; p<3; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = fidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

__device__
inline void get_pos(const int nidx[3], float X[3])
{
  for (int i=0; i<3; i++) 
    X[i] = nidx[i] * cell_lengths[i] + origins[i];
}

__device__
inline void get_mag(float X[3], float A[3])
{
  if (B[1]>0) {
    A[0] = -Kx; 
    A[1] = X[0] * B[2];
    A[2] = -X[0] * B[1];
  } else {
    A[0] = -X[1] * B[2] - Kx;
    A[1] = 0;
    A[2] = X[1] * B[0];
  }
}

__device__
inline bool get_face_values_tet(
    int fid, 
    float X[3][3],
    float A[3][3],
    float rho[3],
    float phi[3],
    float re[3],
    float im[3],
    const float *rho_,
    const float *phi_,
    const float *re_, 
    const float *im_)
{
  int nidxs[3][3], nids[3];
  if (fid2nodes_tet(fid, nidxs)) {
    for (int i=0; i<3; i++) {
      nids[i] = nidx2nid(nidxs[i]);
      re[i] = re_[nids[i]];
      im[i] = im_[nids[i]];
      // rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
      // phi[i] = atan2(im[i], re[i]);
      rho[i] = rho_[nids[i]];
      phi[i] = phi_[nids[i]];
   
      get_pos(nidxs[i], X[i]);
      get_mag(X[i], A[i]); 
    }
    return true;
  }
  else
    return false;
}

__device__
inline int extract_face_tet(
    int fid,
    const float *rho_, 
    const float *phi_,
    const float *re_,
    const float *im_)
{
  const int nnodes = 3;

  float X[nnodes][3], A[nnodes][3], rho[nnodes], phi[nnodes], re[nnodes], im[nnodes];
  get_face_values_tet(fid, X, A, rho, phi, re, im, rho_, phi_, re_, im);

  // compute face shift
  double delta[nnodes], phase_shift = 0;
  for (int i=0; i<nnodes; i++) {
    int j = i % nnodes;
    delta[i] = phi[j] - phi[i]; 
    double li = line_integral(X[i], X[j], A[i], A[j]), 
           qp = 0; // TODO
    delta[i] = mod2pi1(delta[i] - li + qp);
  }

  double critera = phase_shift / (2*M_PI);
  if (fabs(critera)<0.5) return 0; // not punctured

  int chirality = critera>0 ? 1 : -1;
  return chirality;
#if 0
  // gauge transformation
  for (int i=1; i<nnodes; i++) {
    phi[i] = phi[i-1] + delta[i-1];
    re[i] = rho[i] * cos(phi[i]);
    im[i] = phi[i] * sin(phi[i]);
  }

  // TODO: find zero
#endif
}

__global__
static void compute_rho_phi_kernel(
    const float *re,
    const float *im,
    float *rho, 
    float *phi)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i>d[0]*d[1]*d[2]) return;

  rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
  phi[i] = atan2(im[i], re[i]);
}

__global__
static void extract_faces_tet_kernel(
    const float *rho, 
    const float *phi, 
    const float *re,
    const float *im)
{
  unsigned int fid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (fid>d[0]*d[1]*d[2]*12) return;

  extract_face_tet(fid, rho, phi, re, im);
}

void vfgpu_destroy_data()
{
  cudaFree(&d_rho);
  cudaFree(&d_phi);
  cudaFree(&d_re);
  cudaFree(&d_im);

  d_rho = d_phi = d_re = d_im = NULL;
}

void vfgpu_upload_data(
    const int d_[3], 
    const bool pbc_[3], 
    const float origins_[3],
    const float lengths_[3], 
    const float cell_lengths_[3],
    const float B_[3],
    float Kx_,
    const float *re, 
    const float *im)
{
  const int count = d_[0]*d_[1]*d_[2];
  memcpy(dims, d_, sizeof(int)*3);
  
  cudaMemcpyToSymbol(d, d_, sizeof(int)*3);
  cudaMemcpyToSymbol(pbc, pbc_, sizeof(bool)*3);
  cudaMemcpyToSymbol(origins, origins_, sizeof(float)*3);
  cudaMemcpyToSymbol(lengths, lengths_, sizeof(float)*3);
  cudaMemcpyToSymbol(cell_lengths, cell_lengths_, sizeof(float)*3);
  cudaMemcpyToSymbol(B, B_, sizeof(float)*3);
  cudaMemcpyToSymbol(&Kx, &Kx_, sizeof(float));

  if (d_rho == NULL) { // FIXME
    cudaMalloc((void**)&d_rho, sizeof(float)*count);
    cudaMalloc((void**)&d_phi, sizeof(float)*count);
    cudaMalloc((void**)&d_re, sizeof(float)*count);
    cudaMalloc((void**)&d_im, sizeof(float)*count);
  }

  cudaMemcpy(d_re, re, sizeof(float)*count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_im, im, sizeof(float)*count, cudaMemcpyHostToDevice);

  const int nThreadsPerBlock = 256;
  int nBlocks = idivup(count, nThreadsPerBlock);

  compute_rho_phi_kernel<<<nBlocks, nThreadsPerBlock>>>(d_re, d_im, d_rho, d_phi);
}

void vfgpu_extract_faces_tet()
{
  const int count = dims[0]*dims[1]*dims[2]*12; // face counts
  const int nThreadsPerBlock = 256;
  int nBlocks = idivup(count, nThreadsPerBlock);

  fprintf(stderr, "extracting...\n");
  extract_faces_tet_kernel<<<nBlocks, nThreadsPerBlock>>>(d_re, d_im, d_rho, d_phi);
  fprintf(stderr, "finished.\n");
}
