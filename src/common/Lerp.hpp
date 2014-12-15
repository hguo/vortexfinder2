#ifndef _LERP_H
#define _LERP_H

#include "Texel.hpp"
#include <cmath>

template <typename T>
bool lerp2D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars); 

template <typename T>
bool lerp3D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars); 

template <typename T>
bool lerp4D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars); 


////// impl
template <typename T>
bool lerp2D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars)
{
  if (pt[0]<st[0] || pt[0]>=st[0]+sz[0] || 
      pt[1]<st[1] || pt[1]>=st[1]+sz[1]) 
    return false; 

  T p[4]; 
  T x=pt[0] - st[0], 
    y=pt[1] - st[1]; 
  int i = floor(x), 
      j = floor(y); 
  int i1=i+1, j1=j+1; 
  T x0 = i, x1 = i1, 
    y0 = j, y1 = j1; 
  int v; 

  for (v=0; v<num_vars; v++) {
    p[0] = texel2D(ptrs[v], sz, i  , j  );
    p[1] = texel2D(ptrs[v], sz, i1 , j  );
    p[2] = texel2D(ptrs[v], sz, i  , j1 );
    p[3] = texel2D(ptrs[v], sz, i1 , j1 );

    vars[v] = 
        p[0]*(x1-x)*(y1-y)
    	+	p[1]*(x-x0)*(y1-y)
    	+	p[2]*(x1-x)*(y-y0)
    	+	p[3]*(x-x0)*(y-y0); 
  }

  return true; 
}

template <typename T>
bool lerp3D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars)
{
  if (pt[0]<st[0] || pt[0]>=st[0]+sz[0] || 
      pt[1]<st[1] || pt[1]>=st[1]+sz[1] || 
      pt[2]<st[2] || pt[2]>=st[2]+sz[2]) 
    return false; 

  T p[8]; 
  T x=pt[0] - st[0], 
    y=pt[1] - st[1], 
    z=pt[2] - st[2]; 
  int i = floor(x), 
      j = floor(y), 
      k = floor(z); 
  int i1=i+1, j1=j+1, k1=k+1; 
  T x0 = i, x1 = i1, 
        y0 = j, y1 = j1, 
        z0 = k, z1 = k1; 
  int v; 

  for (v=0; v<num_vars; v++) {
    p[0] = texel3D(ptrs[v], sz, i  , j  , k  ); 
    p[1] = texel3D(ptrs[v], sz, i1 , j  , k  ); 
    p[2] = texel3D(ptrs[v], sz, i  , j1 , k  ); 
    p[3] = texel3D(ptrs[v], sz, i1 , j1 , k  ); 
    p[4] = texel3D(ptrs[v], sz, i  , j  , k1 ); 
    p[5] = texel3D(ptrs[v], sz, i1 , j  , k1 ); 
    p[6] = texel3D(ptrs[v], sz, i  , j1 , k1 ); 
    p[7] = texel3D(ptrs[v], sz, i1 , j1 , k1 ); 

    vars[v] = 
        p[0]*(x1-x)*(y1-y)*(z1-z)
    	+	p[1]*(x-x0)*(y1-y)*(z1-z)
    	+	p[2]*(x1-x)*(y-y0)*(z1-z)
    	+	p[3]*(x-x0)*(y-y0)*(z1-z)
    	+	p[4]*(x1-x)*(y1-y)*(z-z0)
    	+	p[5]*(x-x0)*(y1-y)*(z-z0)
    	+	p[6]*(x1-x)*(y-y0)*(z-z0)
    	+	p[7]*(x-x0)*(y-y0)*(z-z0);
  }

  return true; 
}

template <typename T>
bool lerp4D(const T *pt, const int* st, const int* sz, int num_vars, const T **ptrs, T *vars)
{
  if (pt[0]<st[0] || pt[0]>=st[0]+sz[0] || 
      pt[1]<st[1] || pt[1]>=st[1]+sz[1] || 
      pt[2]<st[2] || pt[2]>=st[2]+sz[2] || 
      pt[3]<st[3] || pt[3]>=st[3]+sz[3]) 
    return false; 

  T p[16]; 
  T x=pt[0] - st[0], 
    y=pt[1] - st[1], 
    z=pt[2] - st[2], 
    t=pt[3] - st[3]; 
  int i = floor(x), 
      j = floor(y), 
      k = floor(z), 
      l = floor(t); 
  int i1=i+1, j1=j+1, k1=k+1, l1=l+1; 
  T x0 = i, x1 = i1, 
    y0 = j, y1 = j1, 
    z0 = k, z1 = k1, 
    t0 = l, t1 = l1; 
  int v; // for variables
  int s; // for vertices

  for (v=0; v<num_vars; v++) {
    p[0] = texel4D(ptrs[v], sz, i  , j  , k  , l  ); 
    p[1] = texel4D(ptrs[v], sz, i1 , j  , k  , l  ); 
    p[2] = texel4D(ptrs[v], sz, i  , j1 , k  , l  ); 
    p[3] = texel4D(ptrs[v], sz, i1 , j1 , k  , l  ); 
    p[4] = texel4D(ptrs[v], sz, i  , j  , k1 , l  ); 
    p[5] = texel4D(ptrs[v], sz, i1 , j  , k1 , l  ); 
    p[6] = texel4D(ptrs[v], sz, i  , j1 , k1 , l  ); 
    p[7] = texel4D(ptrs[v], sz, i1 , j1 , k1 , l  ); 
    p[8] = texel4D(ptrs[v], sz, i  , j  , k  , l1 ); 
    p[9] = texel4D(ptrs[v], sz, i1 , j  , k  , l1 ); 
    p[10]= texel4D(ptrs[v], sz, i  , j1 , k  , l1 ); 
    p[11]= texel4D(ptrs[v], sz, i1 , j1 , k  , l1 ); 
    p[12]= texel4D(ptrs[v], sz, i  , j  , k1 , l1 ); 
    p[13]= texel4D(ptrs[v], sz, i1 , j  , k1 , l1 ); 
    p[14]= texel4D(ptrs[v], sz, i  , j1 , k1 , l1 ); 
    p[15]= texel4D(ptrs[v], sz, i1 , j1 , k1 , l1 ); 

    for (s=0; s<16; s++) 
      if (isnan(p[i]) || isinf(p[i]))
        return false; 
  
    vars[v] = 
        p[0]*(x1-x)*(y1-y)*(z1-z)*(t1-t)
    	+	p[1]*(x-x0)*(y1-y)*(z1-z)*(t1-t)
    	+	p[2]*(x1-x)*(y-y0)*(z1-z)*(t1-t)
    	+	p[3]*(x-x0)*(y-y0)*(z1-z)*(t1-t)
    	+	p[4]*(x1-x)*(y1-y)*(z-z0)*(t1-t)
    	+	p[5]*(x-x0)*(y1-y)*(z-z0)*(t1-t)
    	+	p[6]*(x1-x)*(y-y0)*(z-z0)*(t1-t)
    	+	p[7]*(x-x0)*(y-y0)*(z-z0)*(t1-t)
      + p[8]*(x1-x)*(y1-y)*(z1-z)*(t-t0)
    	+	p[9]*(x-x0)*(y1-y)*(z1-z)*(t-t0)
    	+	p[10]*(x1-x)*(y-y0)*(z1-z)*(t-t0)
    	+	p[11]*(x-x0)*(y-y0)*(z1-z)*(t-t0)
    	+	p[12]*(x1-x)*(y1-y)*(z-z0)*(t-t0)
    	+	p[13]*(x-x0)*(y1-y)*(z-z0)*(t-t0)
    	+	p[14]*(x1-x)*(y-y0)*(z-z0)*(t-t0)
    	+	p[15]*(x-x0)*(y-y0)*(z-z0)*(t-t0); 
  }

  return true; 
}

#endif
