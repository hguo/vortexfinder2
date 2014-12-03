#ifndef _UTILS_H
#define _UTILS_H

template <typename T>
inline static T mod2pi(T x)
{
  float y = fmod(x, 2*M_PI); 
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
