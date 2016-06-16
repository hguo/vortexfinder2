#ifndef _INVERSEINTERPOLATION_CUH
#define _INVERSEINTERPOLATION_CUH

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

  T u[3], v[3]; 

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

template <typename T, int meshtype>
__device__
static inline bool find_zero(const T re[], const T im[], const T X[][3], T pos[3], T epsilon=T(0))
{
  if (meshtype == VF2GPU_MESH_TET || meshtype == VF2GPU_MESH_TET) 
    return find_zero_triangle(re, im, X, pos, epsilon);
    // return find_tri_center(X, pos);
  else if (meshtype == VF2GPU_MESH_HEX || meshtype == VF2GPU_MESH_QUAD)
    return find_zero_quad_bilinear(re, im, X, pos, epsilon);
    // return find_quad_center(X, pos);
  else
    return false;
}

#endif
