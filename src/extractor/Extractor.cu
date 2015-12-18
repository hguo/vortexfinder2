#include "Extractor.cuh"
#include "threadIdx.cuh"
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
static unsigned int *d_pfcount;
static gpu_pf_t *d_pfoutput=NULL;
static unsigned int pfcount;
static gpu_pf_t *pfoutput=NULL;

inline void checkCuda(cudaError_t e, const char *situation) {
  if (e != cudaSuccess) {
    printf("CUDA Error: %s: %s\n", situation, cudaGetErrorString(e));
  }
}

inline void checkLastCudaError(const char *situation) {
  checkCuda(cudaGetLastError(), situation);
}

static inline int idivup(int a, int b)
{
  return (a%b!=0) ? (a/b+1) : (a/b); 
}

template <typename T>
__device__
inline bool find_zero_barycentric(const T re[3], const T im[3], T lambda[3], T epsilon=0)
{
  T D = re[0]*im[1] + re[1]*im[2] + re[2]*im[0] - re[2]*im[1] - re[1]*im[0] - re[0]*im[2]; // TODO: check if D=0?
  T det[3] = {
    re[1]*im[2] - re[2]*im[1], 
    re[2]*im[0] - re[0]*im[2], 
    re[0]*im[1] - re[1]*im[0]
  };

  lambda[0] = det[0]/D; 
  lambda[1] = det[1]/D; 
  lambda[2] = det[2]/D;

  // if (lambda[0]>=0 && lambda[1]>=0 && lambda[2]>=0) return true; 
  if (lambda[0]>=-epsilon && lambda[1]>=-epsilon && lambda[2]>=-epsilon) return true; 
  else return false; 
}

template <typename T>
__device__
inline bool find_zero_triangle(const T re[3], const T im[3], const T X[3][3], T pos[3], T epsilon=0)
{
  T lambda[3]; 
  bool succ = find_zero_barycentric(re, im, lambda, epsilon);

  T R[3][2] = {{X[0][0]-X[2][0], X[1][0]-X[2][0]}, 
               {X[0][1]-X[2][1], X[1][1]-X[2][1]}, 
               {X[0][2]-X[2][2], X[1][2]-X[2][2]}}; 

  pos[0] = R[0][0]*lambda[0] + R[0][1]*lambda[1] + X[2][0]; 
  pos[1] = R[1][0]*lambda[0] + R[1][1]*lambda[1] + X[2][1]; 
  pos[2] = R[2][0]*lambda[0] + R[2][1]*lambda[1] + X[2][2]; 

  return succ; 
}

// find the zero point in [0, 1]x[0, 1] quad, using generalized eigenvalue problem
template <typename T>
__device__
static inline bool find_zero_unit_quad_bilinear(const T re[4], const T im[4], T pos[2], T epsilon=0)
{
  T f00 = re[0], f10 = re[1], f01 = re[3], f11 = re[2], // counter-clockwise
    g00 = im[0], g10 = im[1], g01 = im[3], g11 = im[2];
  T A0 = f00 - f10 - f01 + f11, 
    B0 = f10 - f00, 
    C0 = f01 - f00, 
    D0 = f00,
    A1 = g00 - g10 - g01 + g11, 
    B1 = g10 - g00, 
    C1 = g01 - g00, 
    D1 = g00; 
  T M0[4] = {-B0, -D0, -B1, -D1}; // stored in row major
  // T M1[4] = {A0, C0, A1, C1}; // (yM1 - M0)v = 0, v = {x, 1}^T

  T detM1 = A0*C1 - A1*C0; // TODO: check if detM1==0
  T invM1[4] = {C1/detM1, -C0/detM1, -A1/detM1, A0/detM1};

  // Q = invM1*M0
  T Q[4] = {
    invM1[0]*M0[0] + invM1[1]*M0[2], 
    invM1[0]*M0[1] + invM1[1]*M0[3], 
    invM1[2]*M0[0] + invM1[3]*M0[2], 
    invM1[2]*M0[1] + invM1[3]*M0[3]
  };

  // compute y=eig(Q)
  T trace = Q[0] + Q[3];
  T det = Q[0]*Q[3] - Q[1]*Q[2];
  T lambda[2] = {
    trace/2 + sqrt(trace*trace/4 - det), 
    trace/2 - sqrt(trace*trace/4 - det)
  }; 

  T x[2] = {
    (lambda[0]-Q[3])/Q[2], 
    (lambda[1]-Q[3])/Q[2]
  }; 
  T y[2] = {
    lambda[0], 
    lambda[1]
  };

  bool found = false; 
  for (int i=0; i<2; i++) // check the two roots 
    if (x[i]>=0 && x[i]<=1 && y[i]>=0 && y[i]<=1) {
      pos[0] = x[i]; 
      pos[1] = y[i];
      found = true; 
      break; 
    }
  
  if (!found) // check again, loosing creteria
    for (int i=0; i<2; i++)  
      if (x[i]>=-epsilon && x[i]<=1+epsilon && y[i]>=-epsilon && y[i]<=1+epsilon) {
        pos[0] = x[i]; 
        pos[1] = y[i];
        found = true; 
        break; 
      }

  return found; 
}

template <typename T>
__device__
static inline bool find_zero_quad_bilinear(const T re[4], const T im[4], const T X[4][3], T pos[3], T epsilon=0)
{
  T p[2]; 

  bool succ = find_zero_unit_quad_bilinear(re, im, p, epsilon); 
  if (!succ) return false;

  double u[3], v[3]; 

  u[0] = (1-p[0])*X[0][0] + p[0]*X[1][0];
  u[1] = (1-p[0])*X[0][1] + p[0]*X[1][1];
  u[2] = (1-p[0])*X[0][2] + p[0]*X[1][2];

  v[0] = (1-p[0])*X[3][0] + p[0]*X[2][0];
  v[1] = (1-p[0])*X[3][1] + p[0]*X[2][1];
  v[2] = (1-p[0])*X[3][2] + p[0]*X[2][2];

  pos[0] = (1-p[1])*u[0] + p[1]*v[0];
  pos[1] = (1-p[1])*u[1] + p[1]*v[1];
  pos[2] = (1-p[1])*u[2] + p[1]*v[2];

  return true; 
}

template <typename T>
__device__
static inline bool find_tri_center(const T X[3][3], T pos[3])
{
  pos[0] = (X[0][0] + X[1][0] + X[2][0]) / 3;
  pos[1] = (X[0][1] + X[1][1] + X[2][0]) / 3;
  pos[2] = (X[0][2] + X[1][2] + X[2][0]) / 3;

  return true;
}

template <typename T>
__device__
static inline bool find_quad_center(const T X[4][3], T pos[3])
{
  pos[0] = 0.25*(X[0][0] + X[1][0] + X[2][0] + X[3][0]);
  pos[1] = 0.25*(X[0][1] + X[1][1] + X[2][0] + X[3][1]);
  pos[2] = 0.25*(X[0][2] + X[1][2] + X[2][0] + X[3][2]);

  return true;
}

template <typename T, int nnodes>
__device__
static inline bool find_zero(const T re[nnodes], const T im[nnodes], const T X[nnodes][3], T pos[3], T epsilon=T(0))
{
  if (nnodes==3) 
    return find_zero_triangle(re, im, X, pos, epsilon);
    // return find_tri_center(X, pos);
  else if (nnodes==4)
    return find_zero_quad_bilinear(re, im, X, pos, epsilon);
    // return find_quad_center(X, pos);
  else
    return false;
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
inline void fid2fidx_tet(int id, int idx[4])
{
  int nid = id / 12;
  nid2nidx(nid, idx);
  idx[3] = id % 12;
}

__device__
inline void fid2fidx_hex(unsigned int id, int idx[4])
{
  unsigned int nid = id / 3;
  nid2nidx(nid, idx);
  idx[3] = id % 3;
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
bool valid_fidx_hex(const int fidx[4])
{
  if (fidx[3]<0 || fidx[3]>=3) return false;
  else {
    int o[3] = {0}; 
    for (int i=0; i<3; i++) 
      if (pbc[i]) {
        if (fidx[i]<0 || fidx[i]>=d[i]) return false;
      } else {
        if (fidx[i]<0 || fidx[i]>d[i]-1) return false;
        else if (fidx[i] == d[i]-1) o[i] = 1;
      }

    const int sum = o[0] + o[1] + o[2];
    if (sum == 0) return true;
    else if (o[0] + o[1] + o[2] > 1) return false;
    else if (o[0] && fidx[3] == 0) return true; 
    else if (o[1] && fidx[3] == 1) return true;
    else if (o[2] && fidx[3] == 2) return true;
    else return false;
  }
}

__device__
inline void eid2eidx_tet(int id, int idx[4])
{
  int nid = id / 7;
  nid2nidx(nid, idx);
  idx[3] = id % 7;
}

__device__
inline void eid2eidx_hex(unsigned int id, int idx[4]) 
{
  unsigned int nid = id / 3;
  nid2nidx(nid, idx);
  idx[3] = id % 3;
}

__device__
inline bool valid_eidx_tet(const int eidx[4])
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
inline bool valid_eidx_hex(const int eidx[4])
{
  if (eidx[3]<0 || eidx[3]>=3) return false;
  else {
    for (int i=0; i<3; i++) 
      if (pbc[i]) {
        if (eidx[i]<0 || eidx[i]>=d[i]) return false;
      } else {
        if (eidx[i]<0 || eidx[i]>=d[i]-1) return false;
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
inline bool fid2nodes_hex(int id, int nidxs[4][3])
{
  const int nodes_idx[3][4][3] = { // 3 types of faces
    {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, // YZ
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, // ZX
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}  // XY
  };
  
  int fidx[4];
  fid2fidx_hex(id, fidx);

  if (valid_fidx_hex(fidx)) {
    const int type = fidx[3];
    for (int p=0; p<4; p++) 
      for (int q=0; q<3; q++) 
        nidxs[p][q] = fidx[q] + nodes_idx[type][p][q];
    return true;
  }
  else
    return false;
}

template <typename T>
__device__
inline void nidx2pos(const int nidx[3], T X[3])
{
  for (int i=0; i<3; i++) 
    X[i] = nidx[i] * cell_lengths[i] + origins[i];
}

template <typename T>
__device__
inline void magnetic_potential(T X[3], T A[3])
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

template <typename T, int nnodes>
__device__
inline bool get_face_values(
    int fid, 
    T X[nnodes][3],
    T A[nnodes][3],
    T rho[nnodes],
    T phi[nnodes],
    T re[nnodes],
    T im[nnodes],
    const T *rho_,
    const T *phi_,
    const T *re_, 
    const T *im_)
{
  int nidxs[nnodes][3], nids[nnodes];
  bool valid = nnodes == 4 ? fid2nodes_hex(fid, nidxs) : fid2nodes_tet(fid, nidxs);
  
  if (valid) {
    for (int i=0; i<nnodes; i++) {
      nids[i] = nidx2nid(nidxs[i]);
      re[i] = re_[nids[i]];
      im[i] = im_[nids[i]];
      // rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
      // phi[i] = atan2(im[i], re[i]);
      rho[i] = rho_[nids[i]];
      phi[i] = phi_[nids[i]];
   
      nidx2pos(nidxs[i], X[i]);
      magnetic_potential(X[i], A[i]); 
    }
  }

  return valid;
}

template <typename T>
__device__
inline int contour_chirality(
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
    T li = line_integral(X[i], X[j], A[i], A[j]), 
      qp = 0; // TODO
    delta[i] = mod2pi1(delta[i] - li + qp);
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
  for (int i=1; i<nnodes; i++) {
    phi[i] = phi[i-1] + delta[i-1];
    re[i] = rho[i] * cos(phi[i]);
    im[i] = rho[i] * sin(phi[i]);
  }
}

template <typename T, int nnodes>
__device__
inline int extract_face(
    int fid,
    unsigned int *pfcount,
    gpu_pf_t *pfoutput, 
    const T *rho_, 
    const T *phi_,
    const T *re_,
    const T *im_)
{
  T X[nnodes][3], A[nnodes][3], rho[nnodes], phi[nnodes], re[nnodes], im[nnodes];
  T delta[nnodes];
  
  bool valid = get_face_values<T, nnodes>(fid, X, A, rho, phi, re, im, rho_, phi_, re_, im_);
  if (!valid) return 0;

  // compute phase shift
  int chirality = contour_chirality(nnodes, phi, X, A, delta);
  if (chirality == 0) return 0;
  
  // gauge transformation
  gauge_transform(nnodes, rho, delta, phi, re, im);

  // find puncture point
  gpu_pf_t pf; 
  pf.fid = fid;
  pf.chirality = chirality;
  find_zero<T, nnodes>(re, im, X, pf.pos, T(1));
  
  unsigned int idx = atomicInc(pfcount, 0xffffffff);
  pfoutput[idx] = pf;

  return chirality;
}

template <typename T>
__global__
static void compute_rho_phi_kernel(
    T *rho, 
    T *phi,
    const T *re,
    const T *im)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  
  if (i>d[0]*d[1]*d[2]) return;

  rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
  phi[i] = atan2(im[i], re[i]);
}

template <typename T, int nnodes>
__global__
static void extract_faces_kernel(
    unsigned int *pfcount,
    gpu_pf_t *pfoutput,
    const T *rho, 
    const T *phi, 
    const T *re,
    const T *im)
{
  const int ntypes = nnodes == 3 ? 12 : 3;
  int fid = getGlobalIdx_3D_1D();
  if (fid>d[0]*d[1]*d[2]*ntypes) return;

#if 0 // use global memory
  extract_face<T, nnodes>(fid, pfcount, pfoutput, rho, phi, re, im);
#else // use shared memory
  extern __shared__ char smem[];
  unsigned int *spfcount = (unsigned int*)smem;
  gpu_pf_t *spfoutput = (gpu_pf_t*)(smem + sizeof(int));
 
  if (threadIdx.x == 0)
    *spfcount = 0;
  __syncthreads();
  
  extract_face<T, nnodes>(fid, spfcount, spfoutput, rho, phi, re, im);
  __syncthreads();

  if (threadIdx.x == 0 && (*spfcount)>0) {
    unsigned int idx = atomicAdd(pfcount, *spfcount);
    // printf("idx=%d, count=%d\n", idx, *spfcount);
    memcpy(pfoutput + idx, spfoutput, (*spfcount) * sizeof(gpu_pf_t));
  }
#endif
}

void vfgpu_destroy_data()
{
  cudaFree(d_rho);
  cudaFree(d_phi);
  cudaFree(d_re);
  cudaFree(d_im);
  cudaFree(d_pfoutput);

  free(pfoutput);
  
  checkLastCudaError("destroying data");

  d_rho = d_phi = d_re = d_im = NULL; 
  d_pfoutput = NULL;
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
  const int max_pf_count = count*12*0.1;
  memcpy(dims, d_, sizeof(int)*3);
  
  checkLastCudaError("dummy");
  
  cudaMemcpyToSymbol(d, d_, sizeof(int)*3);
  cudaMemcpyToSymbol(pbc, pbc_, sizeof(bool)*3);
  cudaMemcpyToSymbol(origins, origins_, sizeof(float)*3);
  cudaMemcpyToSymbol(lengths, lengths_, sizeof(float)*3);
  cudaMemcpyToSymbol(cell_lengths, cell_lengths_, sizeof(float)*3);
  cudaMemcpyToSymbol(B, B_, sizeof(float)*3);
  cudaMemcpyToSymbol(Kx, &Kx_, sizeof(float));

  checkLastCudaError("copy to symbols");

  if (d_rho == NULL) { // FIXME
    cudaMalloc((void**)&d_re, sizeof(float)*count);
    cudaMalloc((void**)&d_im, sizeof(float)*count);
    cudaMalloc((void**)&d_rho, sizeof(float)*count);
    cudaMalloc((void**)&d_phi, sizeof(float)*count);
    cudaMalloc((void**)&d_pfcount, sizeof(unsigned int));
    cudaMalloc((void**)&d_pfoutput, sizeof(gpu_pf_t)*max_pf_count);

    pfoutput = (gpu_pf_t*)malloc(max_pf_count*sizeof(gpu_pf_t));
  }

  cudaMemcpy(d_re, re, sizeof(float)*count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_im, im, sizeof(float)*count, cudaMemcpyHostToDevice);
  
  checkLastCudaError("copy data to device");

  const int nThreadsPerBlock = 128;
  int nBlocks = idivup(count, nThreadsPerBlock);

  compute_rho_phi_kernel<<<nBlocks, nThreadsPerBlock>>>(d_rho, d_phi, d_re, d_im);
  cudaDeviceSynchronize();
  checkLastCudaError("compute rho and phi");
}

void vfgpu_extract_faces(int *pfcount_, gpu_pf_t **pfbuf, int discretization)
{
  const int nnodes = discretization == GLGPU3D_MESH_HEX ? 4 : 3;
  const int nfacetypes = nnodes == 3 ? 12 : 3;

  const int threadCount = dims[0]*dims[1]*dims[2]*nfacetypes; // face counts
  const int maxGridDim = 1024; // 32768;
  const int blockSize = 256;
  const int nBlocks = idivup(threadCount, blockSize);
  dim3 gridSize; 
  if (nBlocks >= maxGridDim) 
    gridSize = dim3(idivup(nBlocks, maxGridDim), maxGridDim);
  else 
    gridSize = dim3(nBlocks);
  const int sharedSize = blockSize * sizeof(gpu_pf_t) + sizeof(unsigned int);
    
  cudaMemset(d_pfcount, 0, sizeof(unsigned int));
  checkLastCudaError("extract faces [0]");
  fprintf(stderr, "gridSize={%d, %d, %d}, blockSize=%d\n", gridSize.x, gridSize.y, gridSize.z, blockSize);
  if (nnodes == 3)
    extract_faces_kernel<float, 3><<<gridSize, blockSize, sharedSize>>>(d_pfcount, d_pfoutput, d_rho, d_phi, d_re, d_im);
  else if (nnodes == 4)
    extract_faces_kernel<float, 4><<<gridSize, blockSize, sharedSize>>>(d_pfcount, d_pfoutput, d_rho, d_phi, d_re, d_im);
  checkLastCudaError("extract faces [1]");
 
  cudaMemcpy((void*)&pfcount, d_pfcount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("pfcount=%d\n", pfcount);
  if (pfcount>0)
    cudaMemcpy(pfoutput, d_pfoutput, sizeof(gpu_pf_t)*pfcount, cudaMemcpyDeviceToHost);
  checkLastCudaError("extract faces [2]");
  
  cudaDeviceSynchronize();

  *pfcount_ = pfcount;
  *pfbuf = pfoutput;
}
