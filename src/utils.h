#ifndef _UTILS_H
#define _UTILS_H

inline static float mod2pi(float x)
{
  float y = fmod(x, 2*M_PI); 
  if (y<0) y+= 2*M_PI;
  return y; 
}

static inline bool find_zero_triangle(float r[3], float i[3], float lambda[3])
{
  float D = r[0]*i[1] + r[1]*i[2] + r[2]*i[0] - r[2]*i[1] - r[1]*i[0] - r[0]*i[2]; // TODO: check if D=0?
  float det[3] = {
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

static inline bool find_zero_triangle(float r[3], float i[3], float x[3], float y[3], float z[3], float pos[3])
{
  float lambda[3]; 
  if (!find_zero_triangle(r, i, lambda)) return false; 

  float T[3][2] = {{x[0]-x[2], x[1]-x[2]}, 
                   {y[0]-y[2], y[1]-y[2]}, 
                   {z[0]-z[2], z[1]-z[2]}}; 

  pos[0] = T[0][0]*lambda[0] + T[0][1]*lambda[1] + x[2]; 
  pos[1] = T[1][0]*lambda[0] + T[1][1]*lambda[1] + y[2]; 
  pos[2] = T[2][0]*lambda[0] + T[2][1]*lambda[1] + z[2]; 

  return true; 
}

static inline float gauge_transformation(const float *x0, const float *x1, float Kex, const float *B)
{
  float gx, gy, gz; 
  float dx[3] = {x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]};
 
  float x = x0[0] + dx[0]*0.5f, 
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
