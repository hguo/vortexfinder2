#ifndef _TEXEL_H
#define _TEXEL_H

template <typename T>
const T& texel2D(const T* p, const int* sz, int x, int y); 

template <typename T>
T& texel2D(T* p, const int* sz, int x, int y); 

template <typename T>
const T& texel3D(const T* p, const int* sz, int x, int y, int z); 

template <typename T>
T& texel3D(T* p, const int* sz, int x, int y, int z); 

template <typename T>
const T& texel4D(const T* p, const int* sz, int x, int y, int z, int t); 

template <typename T>
T& texel4D(T* p, const int* sz, int x, int y, int z, int t); 



////// impl  
template <typename T>
const T& texel2D(const T* p, const int* sz, int x, int y) 
{
  return p[x + sz[0]*y]; 
}

template <typename T>
T& texel2D(T* p, const int* sz, int x, int y) 
{
  return p[x + sz[0]*y]; 
}

template <typename T>
const T& texel3D(const T* p, const int* sz, int x, int y, int z)
{
  return p[x + sz[0]*(y + sz[1]*z)]; 
}

template <typename T>
T& texel3D(T* p, const int* sz, int x, int y, int z)
{
  return p[x + sz[0]*(y + sz[1]*z)]; 
}

template <typename T>
const T& texel4D(const T* p, const int* sz, int x, int y, int z, int t)
{
  return p[x + sz[0]*(y + sz[1]*(z + sz[2]*t))]; 
}

template <typename T>
T& texel4D(T* p, const int* sz, int x, int y, int z, int t)
{
  return p[x + sz[0]*(y + sz[1]*(z + sz[2]*t))]; 
}

#endif
