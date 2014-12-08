#ifndef _UTILS_H
#define _UTILS_H

template <typename T>
inline static T mod2pi(T x)
{
  T y = fmod(x, 2*M_PI); 
  if (y<0) y+= 2*M_PI;
  return y; 
}

template <typename T>
static inline bool find_zero_triangle(T r[3], T i[3], T lambda[3])
{
  T D = r[0]*i[1] + r[1]*i[2] + r[2]*i[0] - r[2]*i[1] - r[1]*i[0] - r[0]*i[2]; // TODO: check if D=0?
  T det[3] = {
    r[1]*i[2] - r[2]*i[1], 
    r[2]*i[0] - r[0]*i[2], 
    r[0]*i[1] - r[1]*i[0]
  };

  lambda[0] = det[0]/D; 
  lambda[1] = det[1]/D; 
  lambda[2] = det[2]/D; 
  
  if (lambda[0]>=0 && lambda[1]>=0 && lambda[2]>=0) return true; 
  else return false; 
}

template <typename T>
static inline bool find_zero_triangle(T r[3], T i[3], T X0[3], T X1[3], T X2[3], T pos[3])
{
  T lambda[3]; 
  if (!find_zero_triangle(r, i, lambda)) return false; 

  T R[3][2] = {{X0[0]-X2[0], X1[0]-X2[0]}, 
               {X0[1]-X2[1], X1[1]-X2[1]}, 
               {X0[2]-X2[2], X1[2]-X2[2]}}; 

  pos[0] = R[0][0]*lambda[0] + R[0][1]*lambda[1] + X2[0]; 
  pos[1] = R[1][0]*lambda[0] + R[1][1]*lambda[1] + X2[1]; 
  pos[2] = R[2][0]*lambda[0] + R[2][1]*lambda[1] + X2[2]; 

  return true; 
}

// find the zero point in [0, 1]x[0, 1] quad, using generalized eigenvalue problem
template <typename T>
static inline bool find_zero_quad(T re[4], T im[4], T pos[2])
{
  const T epsilon = 0.01; 
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

  // if (!found) 
  //   fprintf(stderr, "roots not found: {%f, %f}, {%f, %f}\n", x[0], y[0], x[1], y[1]);
  return found; 
}

template <typename T>
static inline float gauge_transformation(const T *x0, const T *x1, T Kex, const T *B)
{
  T gx, gy, gz; 
  T dx[3] = {x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]};
 
  T x = x0[0] + dx[0]*0.5f, 
    y = x0[1] + dx[1]*0.5f, 
    z = x0[2] + dx[2]*0.5f; 

  if (B[1]>0) { // Y-Z gauge
    gx = dx[0] * Kex; 
    gy =-dx[1] * x * B[2]; // -dy*x^hat*Bz
    gz = dx[2] * x * B[1]; //  dz*x^hat*By
  } else { // X-Z gauge
    gx = dx[0] * y * B[2]  //  dx*y^hat*Bz
        +dx[0] * Kex; 
    gy = 0; 
    gz =-dx[2] * y * B[0]; // -dz*y^hat*Bx
  }

  return gx + gy + gz; 
}

#endif
