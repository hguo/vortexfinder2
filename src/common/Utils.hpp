#ifndef _UTILS_H
#define _UTILS_H

#include <cmath>

template <typename T>
inline static T fmod1(T x, T y)
{
  T z = fmod(x, y);
  if (z<0) z += y;
  return z;
}

template <typename T>
inline static T mod2pi(T x)
{
  T y = fmod(x, 2*M_PI); 
  if (y<0) y+= 2*M_PI;
  return y; 
}

template <typename T>
inline static T mod2pi1(T x)
{
  return mod2pi(x + M_PI) - M_PI;
}

template <typename T> int sgn(T x) {
  return (T(0) < x) - (x < T(0));
}

template <typename T>
static inline void cross_product(const T A[3], const T B[3], T C[3])
{
  C[0] = A[1]*B[2] - A[2]*B[1]; 
  C[1] = A[2]*B[0] - A[0]*B[2]; 
  C[2] = A[0]*B[1] - A[1]*B[0];
}

template <typename T>
static inline T area(const T A[3], const T B[3], const T C[3])
{
  T AB[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]}, 
    AC[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};

  T D[3];
  cross_product(AB, AC, D);
  T length = sqrt(D[0]*D[0] + D[1]*D[1] + D[2]*D[2]);

  return 0.5 * length;
}

template <typename T>
static inline void normalize(T X[3])
{
  T length = sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);

  if (length>0) {
    X[0] /= length;
    X[1] /= length;
    X[2] /= length;
  }
}

template <typename T>
static inline T dist(const T A[3], const T B[3])
{
  T d[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
  return sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}

template <typename T>
static inline T inner_product(const T A[3], const T B[3])
{
  return A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
}

#endif
