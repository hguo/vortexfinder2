/*
PKUVIS CONFIDENTIAL
___________________

Copyright (c) 2009-2012, PKU Visualization and Visual Analytics Group 
Produced at Peking University, Beijing, China.
All rights reserved.
                                                                             
NOTICE: THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF  VISUALIZATION 
AND VISUAL ANALYTICS GROUP (PKUVIS), PEKING UNIVERSITY. DISSEMINATION
OF  THIS  INFORMATION  OR  REPRODUCTION OF THIS  MATERIAL IS STRICTLY 
FORBIDDEN UNLESS PRIOR WRITTEN PERMISSION IS OBTAINED FROM PKUVIS.
*/

#include "cutil/cutil_math.h"
#include <cstdio>

#ifndef cudaTextureType3D
 #define cudaTextureType3D 3
#endif

const float pi = 3.14159f; 

inline void checkCuda(cudaError_t e, const char *situation) {
    if (e != cudaSuccess) {
        printf("CUDA Error: %s: %s\n", situation, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError(const char *situation) {
    checkCuda(cudaGetLastError(), situation);
}

inline int iDivUp(int a, int b)
{
    return (a%b!=0)?(a/b+1):(a/b);
}

inline __host__ __device__ float3 operator*(float3 a, int3 b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); 
}

inline __host__ __device__ void operator*=(float3 &a, int3 b)
{
    a.x*=b.x; a.y*=b.y; a.z*=b.z; 
}

inline __host__ __device__ bool operator<(float3 a, float3 b)
{
    return a.x<b.x && a.y<b.y && a.z<b.z; 
}

inline __host__ __device__ bool operator>(float3 a, float3 b)
{
    return a.x>b.x && a.y>b.y && a.z>b.z; 
}

inline __host__ __device__ bool operator<=(float3 a, float3 b)
{
    return a.x<=b.x && a.y<=b.y && a.z<=b.z; 
}

inline __host__ __device__ bool operator>=(float3 a, float3 b)
{
    return a.x>=b.x && a.y>=b.y && a.z>=b.z; 
}

__device__ __host__
inline void mulmat(const float a[16], const float b[16], float r[16])
{
    int i, j;

    for (i = 0; i < 4; i++) {
    	for (j = 0; j < 4; j++) {
    	    r[i*4+j] = 
    		a[i*4+0]*b[0*4+j] +
    		a[i*4+1]*b[1*4+j] +
    		a[i*4+2]*b[2*4+j] +
    		a[i*4+3]*b[3*4+j];
    	}
    }
}

__device__ __host__
inline float4 mulmatvec(const float m[16], float4 vec)
{
    float4 rtn; 

    rtn.x = m[0]*vec.x + m[4]*vec.y + m[8]*vec.z + m[12]*vec.w; 
    rtn.y = m[1]*vec.x + m[5]*vec.y + m[9]*vec.z + m[13]*vec.w; 
    rtn.z = m[2]*vec.x + m[6]*vec.y +m[10]*vec.z + m[14]*vec.w; 
    rtn.w = m[3]*vec.x + m[7]*vec.y +m[11]*vec.z + m[15]*vec.w; 

    return rtn; 
}

__device__ __host__
inline void mulmatvec(const float matrix[16], const float in[4], float out[4])
{
    int i;

    for (i=0; i<4; i++) {
	out[i] = 
	    in[0] * matrix[0*4+i] +
	    in[1] * matrix[1*4+i] +
	    in[2] * matrix[2*4+i] +
	    in[3] * matrix[3*4+i];
    }
}

__device__ __host__
inline float3 transform(const float matrix[16], const float3 in)
{
#if 0
    float4 in_ = make_float4(in, 1); 
    float3 out; 

    out.x = dot(make_float4(matrix[0], matrix[4], matrix[8], matrix[12]), in_); 
    out.y = dot(make_float4(matrix[1], matrix[5], matrix[9], matrix[13]), in_); 
    out.z = dot(make_float4(matrix[2], matrix[6], matrix[10], matrix[14]), in_); 

    return out; 
#else
    float in_[4], out_[4]; 
    in_[0] = in.x; in_[1] = in.y; in_[2] = in.z; in_[3] = 1; 

    mulmatvec(matrix, in_, out_); 
    return make_float3(out_[0], out_[1], out_[2]); 
#endif
}

__device__ __host__
inline bool invmat(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;

    inv[0] =   m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
             + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] =  -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
             - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] =   m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
             + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
             - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] =  -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
             - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] =   m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
             + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] =  -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
             - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
             + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] =   m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
             + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] =  -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
             - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
             + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
             - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] =  -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
             - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] =   m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
             + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
             - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
             + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0)
        return false;

    det = 1.0f / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}


__device__ __host__
inline bool unproject(float3 win, const float *mvmatrix, const float *projmatrix, int *viewport, float3 &obj)
{
    float  finalMatrix[16];
    float  in[4], out[4];

    mulmat(mvmatrix, projmatrix, finalMatrix);
    if (!invmat(finalMatrix, finalMatrix)) return false;

    in[0] = (win.x - viewport[0]) * 2.f / viewport[2] - 1.f;
    in[1] = (win.y - viewport[1]) * 2.f / viewport[3] - 1.f;
    in[2] = 2.f * win.z - 1.f;
    in[3] = 1.f;

    mulmatvec(finalMatrix, in, out);
    if (out[3] == 0.f) return false;

    obj = make_float3(out[0], out[1], out[2]) / out[3];
    return true;
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__ __host__
inline bool intersectBox(float3 rayO, float3 rayD, float &tnear, float &tfar, 
    const float3 boxmin=make_float3(0.f), 
    const float3 boxmax=make_float3(1.f))
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.f) / rayD;
    float3 tbot = invR * (boxmin - rayO);
    float3 ttop = invR * (boxmax - rayO);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	tnear = largest_tmin;
	tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// intersect ray with a sphere. 
// notice: make sure rayD is normalized.
__device__ __host__
inline bool intersectSphere(float3 rayO, float3 rayD, float &tnear, float &tfar, 
    const float radius=1.f, const float3 center=make_float3(0.f))
{
    float3 offset = rayO - center; 
    float B = 2 * dot(rayD, offset), 
          C = dot(offset, offset) - radius*radius; 
    float D = B*B - 4*C; 

    if (D >= 0.f) {
        tnear = 0.5f * (-B-sqrtf(D)); 
        tfar  = 0.5f * (-B+sqrtf(D)); 
        return true; 
    } else 
        return false; 
}

__device__ __host__
inline bool intersectGeoRange(float3 rayO, float3 rayD, 
    float west, float east, float north, float south, 
    float &tnear, float &tfar, 
    const float radius=1.f, const float3 center=make_float3(0.f))
{
    // TODO
    return false; 
}

__device__ __host__
inline float3 castersianToSpherical(float3 v)
{
    float r     = length(v), 
          lon   = atan2f(v.y, v.x), 
          lat   = asinf(v.z / r); 

    lon = lon<0 ? lon+2*pi : lon; 

    return make_float3(r, lon, lat); 
}

__device__ __host__
inline bool insideBox(float3 v, 
    float3 boxmin=make_float3(0.f), 
    float3 boxmax=make_float3(1.f))
{
    return v<=boxmax && v>=boxmin; 
}

__device__ __host__
inline float3 sphericalToTexture(float3 v, float radius0, float radius1, 
                                           float lon0=-pi, float lon1=pi, 
                                           float lat0=-pi/2, float lat1=pi/2)
{
    float z = (v.x-radius0) / (radius1-radius0), // TODO: performance opt
          x = (v.y-lon0) / (lon1-lon0), 
          y = (v.z-lat0) / (lat1-lat0); 

    return make_float3(x, y, z); 
}

__device__ __host__
inline bool clipping(float3 rayO, float3 rayD, float &tnear, float &tfar, float4 plane)
{
    float3 p0 = rayO + rayD*tnear, 
           p1 = rayO + rayD*tfar; 
    float  f0 = dot(plane, make_float4(p0, 1)), 
           f1 = dot(plane, make_float4(p1, 1)); 
    
    if (f0>0 && f1>0) return false; 
    if (f0<0 && f1<0) return true; 

    float t = -dot(plane, make_float4(rayO, 1)) / (dot(plane, make_float4(rayD, 0)) + 1e-6f); 

    if (f0>0 && f1<0) tnear = max(t, tnear);
    else tfar = min(t, tfar); 

    return true; 
}

template<class DataType, enum cudaTextureReadMode readMode>
__device__
inline float3 gradient(texture<DataType, cudaTextureType3D, readMode> texRef, float3 coords, float delta=1)
{
    float3 sample1, sample2; 
    float3 dx = make_float3(delta, 0.f, 0.f), 
           dy = make_float3(0.f, delta, 0.f), 
           dz = make_float3(0.f, 0.f, delta); 

    sample1.x = tex3D(texRef, coords.x-delta, coords.y, coords.z); 
    sample2.x = tex3D(texRef, coords.x+delta, coords.y, coords.z); 
    sample1.y = tex3D(texRef, coords.x, coords.y-delta, coords.z); 
    sample2.y = tex3D(texRef, coords.x, coords.y+delta, coords.z); 
    sample1.z = tex3D(texRef, coords.x, coords.y, coords.z-delta); 
    sample2.z = tex3D(texRef, coords.x, coords.y, coords.z+delta); 

    return (sample2-sample1) / delta; 
}

__device__ __host__
inline float3 phong(float3 N, float3 V, float3 L, float3 Ka, float3 Kd, float3 Ks, float shiness)
{
    float3 lightColor = make_float3(1.f), 
           ambientColor = make_float3(1.f); 

    if (length(N)<1e-5) return make_float3(0.f); 
    N = normalize(N); 

    float3 H = normalize(L+V); // half way
    float3 ambient = Ka * ambientColor; // ambient
    float3 diffuse = Kd * lightColor * max(dot(L, N), 0.f); // diffuse
    float3 specular= Ks * lightColor * pow(max(dot(H, N), 0.f), shiness); 

    return ambient + diffuse + specular; 
}

__device__ __host__
inline float3 cook(float3 N, float3 V, float3 L, float3 Ka, float3 Kd, float3 Ks) 
{
    float mean  = 0.7f; 
    float scale = 0.2f; 

    if (length(N)<1e-5) return make_float3(0.f); 
    N = normalize(N); 

    float3 lightIntensity = make_float3(1.f); 

    float3 H = normalize(L + V); 
    float n_h = dot(N, H), 
          n_v = dot(N, V) + 1e-6f, 
          v_h = dot(V, H) + 1e-6f, 
          n_l = dot(N, L); 

    float3 diffuse = Kd * max(n_l, 0.f); 
    float  fresnel = pow(1.f + v_h, 4.f); 
    float  delta   = acos(n_h); 
    float  exponent= -pow((delta/mean), 2.f); 
    float  microfacets = scale * exp(exponent);

    float  term1 = 2 * n_h * n_v / v_h, 
           term2 = 2 * n_h * n_l / v_h; 
    float  selfshadow = min(1.f, min(term1, term2)); 

    float3 specular = Ks * fresnel * microfacets * selfshadow / n_v; 
    
    return clamp(lightIntensity * (diffuse + specular), 0.f, 1.f); 
}
