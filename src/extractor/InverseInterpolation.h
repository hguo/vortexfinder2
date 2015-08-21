#ifndef _INVERSE_INTERPOLATION_H
#define _INVERSE_INTERPOLATION_H

#include <cmath>

template <typename T>
static inline bool find_zero_barycentric(const T re[3], const T im[3], T lambda[3], T epsilon=0)
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
static inline bool find_zero_triangle(const T re[3], const T im[3], const T X[3][3], T pos[3], T epsilon=0)
{
  T lambda[3]; 
  if (!find_zero_barycentric(re, im, lambda, epsilon)) return false; 

  T R[3][2] = {{X[0][0]-X[2][0], X[1][0]-X[2][0]}, 
               {X[0][1]-X[2][1], X[1][1]-X[2][1]}, 
               {X[0][2]-X[2][2], X[1][2]-X[2][2]}}; 

  pos[0] = R[0][0]*lambda[0] + R[0][1]*lambda[1] + X[2][0]; 
  pos[1] = R[1][0]*lambda[0] + R[1][1]*lambda[1] + X[2][1]; 
  pos[2] = R[2][0]*lambda[0] + R[2][1]*lambda[1] + X[2][2]; 

  return true; 
}

// find the zero point in [0, 1]x[0, 1] quad, using generalized eigenvalue problem
template <typename T>
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
  T M0[4] = {-B0, -D0, -B1, -D1}, // stored in row major
    M1[4] = {A0, C0, A1, C1}; // (yM1 - M0)v = 0, v = {x, 1}^T

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

  T xx, yy;
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
static inline bool find_quad_center(const T X[4][3], T pos[3])
{
  pos[0] = 0.25*(X[0][0] + X[1][0] + X[2][0] + X[3][0]);
  pos[1] = 0.25*(X[0][1] + X[1][1] + X[2][0] + X[3][1]);
  pos[2] = 0.25*(X[0][2] + X[1][2] + X[2][0] + X[3][2]);

  return true;
}

template <typename T>
static inline bool find_tri_center(const T X[3][3], T pos[3])
{
  pos[0] = (X[0][0] + X[1][0] + X[2][0]) / 3;
  pos[1] = (X[0][1] + X[1][1] + X[2][0]) / 3;
  pos[2] = (X[0][2] + X[1][2] + X[2][0]) / 3;

  return true;
}

template <typename T>
static inline bool find_zero_quad_barycentric(const T R[4], const T I[4], const T X[4][3], T pos[3], T epsilon=0)
{
  T X0[3][3] = {{X[0][0], X[0][1], X[0][2]}, 
                {X[1][0], X[1][1], X[1][2]}, 
                {X[2][0], X[2][1], X[2][2]}}, 
    X1[3][3] = {{X[0][0], X[0][1], X[0][2]}, 
                {X[2][0], X[2][1], X[2][2]}, 
                {X[3][0], X[3][1], X[3][2]}}; 
  T R0[3] = {R[0], R[1], R[2]}, 
    R1[3] = {R[0], R[2], R[3]}; 
  T I0[3] = {I[0], I[1], I[2]}, 
    I1[3] = {I[0], I[2], I[3]}; 

  if (find_zero_triangle(R0, I0, X0, pos)) 
    return true; 
  else if (find_zero_triangle(R1, I1, X1, pos)) 
    return true;
  if (find_zero_triangle(R0, I0, X0, pos, epsilon))
    return true;
  else if (find_zero_triangle(R1, I1, X1, pos, epsilon))
    return true;
  else return false;
}

template <typename T>
static inline bool find_zero_unit_quad_barycentric(const T R[4], const T I[4], T pos[2], T epsilon=0)
{
  T X0[3][3] = {{0, 0, 0}, 
                {1, 0, 0},
                {0, 1, 0}};
  T X1[3][3] = {{1, 0, 0}, 
                {1, 1, 0},
                {0, 1, 0}};
  T R0[3] = {R[0], R[1], R[2]}, 
    R1[3] = {R[0], R[2], R[3]}; 
  T I0[3] = {I[0], I[1], I[2]}, 
    I1[3] = {I[0], I[2], I[3]};
  T p[3];

  if (find_zero_triangle(R0, I0, X0, p)) {
    pos[0] = p[0]; 
    pos[1] = p[1];
    return true;
  } else 
    return false;
}

template <typename T>
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
static inline bool find_zero_linear(T f0, T f1, const T X0[3], const T X1[3], T p[3])
{
  // (1-alpha)*f0 + alpha*f1 = 0
  if (f0 - f1 == 0) return false;
  
  T alpha = f0 / (f0 - f1);
  if (alpha<0 || alpha>=1) return false;

  p[0] = (1-alpha)*X0[0] + alpha*X1[0];
  p[1] = (1-alpha)*X0[1] + alpha*X1[1];
  p[2] = (1-alpha)*X0[2] + alpha*X1[2];

  return true;
}

template <typename T>
static inline bool line_cross(const T P0[3], const T P1[3], const T Q0[3], const T Q1[3], T pos[3])
{
  // TODO
  T da[3] = {P1[0] - P0[0], P1[1] - P0[1], P1[2] - P0[2]}, 
    db[3] = {Q1[0] - Q0[0], Q1[1] - Q0[1], Q1[2] - Q0[2]}, 
    dc[3] = {Q0[0] - P0[0], Q0[1] - P0[1], Q0[2] - P0[2]};

  // if (dot(dc, cross(da, db)) != 0.0) return false; // not coplanar
  return false;
}

template <typename T>
static inline bool find_zero_quad_line_cross(const T re[4], const T im[4], const T X[4][3], T pos[3], T epsilon=0)
{
  bool cr[4], ci[4]; 
  T pr0[4][3], pi0[4][3];
  T *pr[2], *pi[2];
  
  cr[0] = find_zero_linear(re[0], re[1], X[0], X[1], pr0[0]);
  cr[1] = find_zero_linear(re[1], re[2], X[1], X[2], pr0[1]); 
  cr[2] = find_zero_linear(re[2], re[3], X[2], X[3], pr0[2]); 
  cr[3] = find_zero_linear(re[3], re[0], X[3], X[0], pr0[3]);
  
  ci[0] = find_zero_linear(im[0], im[1], X[0], X[1], pi0[0]);
  ci[1] = find_zero_linear(im[1], im[2], X[1], X[2], pi0[1]); 
  ci[2] = find_zero_linear(im[2], im[3], X[2], X[3], pi0[2]); 
  ci[3] = find_zero_linear(im[3], im[0], X[3], X[0], pi0[3]);

  int crs = cr[0] + cr[1] + cr[2] + cr[3], 
      cis = ci[0] + ci[1] + ci[2] + ci[3];
  if (crs!=2 || cis!=2) return false;

  for (int i=0, k=0; i<4; i++) 
    if (cr[i]) pr[k++] = pr0[i]; 
  
  for (int i=0, k=0; i<4; i++) 
    if (ci[i]) pi[k++] = pi0[i]; 

  return line_cross(pr[0], pr[1], pi[0], pi[1], pos);
}

#endif
