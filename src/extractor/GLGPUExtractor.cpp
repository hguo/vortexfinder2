#include <cmath>
#include <cassert>
#include "io/GLGPUDataset.h"
#include "GLGPUExtractor.h"
#include "Utils.h"

enum {
  FACE_XY = 0, 
  FACE_YZ = 1, 
  FACE_XZ = 2
};

enum {
  TRIANGLE_UPPER = 0, 
  TRIANGLE_LOWER = 1
}; 

GLGPUVortexExtractor::GLGPUVortexExtractor()
{
}

GLGPUVortexExtractor::~GLGPUVortexExtractor()
{
}
  
void GLGPUVortexExtractor::SetDataset(const GLDataset *ds)
{
  _ds = (const GLGPUDataset*)ds; 
}

void GLGPUVortexExtractor::solve(int x, int y, int z, int face)
{
  float re[4], im[4], amp[4], phase[4], dirs[4];
  int X[4][3]; 
  float flux = 0.f; 
  switch (face) {
  case FACE_XY: 
    X[0][0] = x;   X[0][1] = y;   X[0][2] = z; 
    X[1][0] = x+1; X[1][1] = y;   X[1][2] = z; 
    X[2][0] = x+1; X[2][1] = y+1; X[2][2] = z; 
    X[3][0] = x;   X[3][1] = y+1; X[3][2] = z;
    flux = _ds->cellLengths()[0] * _ds->cellLengths()[1] * _ds->B()[2]; 
    break; 
  
  case FACE_YZ: 
    X[0][0] = x;   X[0][1] = y;   X[0][2] = z; 
    X[1][0] = x;   X[1][1] = y+1; X[1][2] = z; 
    X[2][0] = x;   X[2][1] = y+1; X[2][2] = z+1; 
    X[3][0] = x;   X[3][1] = y;   X[3][2] = z+1; 
    flux = _ds->cellLengths()[0] * _ds->cellLengths()[2] * _ds->B()[1]; 
    break; 
  
  case FACE_XZ: 
    X[0][0] = x;   X[0][1] = y;   X[0][2] = z; 
    X[1][0] = x+1; X[1][1] = y;   X[1][2] = z; 
    X[2][0] = x+1; X[2][1] = y;   X[2][2] = z+1; 
    X[3][0] = x;   X[3][1] = y;   X[3][2] = z+1; 
    flux = _ds->cellLengths()[1] * _ds->cellLengths()[2] * _ds->B()[0]; 
    break; 

  default: assert(0); break;  
  }

  for (int i=0; i<4; i++) {
    re[i] = _ds->re(X[i][0], X[i][1], X[i][2]); 
    im[i] = _ds->im(X[i][0], X[i][1], X[i][2]); 
    amp[i] = _ds->amp(X[i][0], X[i][1], X[i][2]); 
    phase[i] = atan2(im[i], re[i]);
  }

#if 1 // gauge
  float delta[4] = {
    phase[1] - phase[0] + gauge(X[0], X[1]), 
    phase[2] - phase[1] + gauge(X[1], X[2]),  
    phase[3] - phase[2] + gauge(X[2], X[3]), 
    phase[0] - phase[3] + gauge(X[3], X[0])
  };
#else
  float delta[4] = {
    phase[1] - phase[0], 
    phase[2] - phase[1],  
    phase[3] - phase[2], 
    phase[0] - phase[3]
  };
#endif

  float sum = 0.f;
  float delta1[4]; 
  for (int i=0; i<4; i++) {
    delta1[i] = mod2pi(delta[i] + M_PI) - M_PI;
    sum += delta1[i]; 
  }
  sum += flux; 

#if 1 //gauge
  phase[1] = phase[0] + delta1[0]; 
  phase[2] = phase[1] + delta1[1]; 
  phase[3] = phase[2] + delta1[2];
  
  for (int i=0; i<4; i++) {
    re[i] = amp[i] * cos(phase[i]); 
    im[i] = amp[i] * sin(phase[i]); 
  }
#endif

  float ps = sum / (2*M_PI);
  if (fabs(ps)>0.99f) {
#if 0 // barycentric
    float re0[3] = {re[1], re[2], re[0]},
          im0[3] = {im[1], im[2], im[0]},
          re1[3] = {re[3], re[2], re[0]},
          im1[3] = {im[3], im[2], im[0]};
    float lambda[3];

    bool upper = find_zero_triangle(re0, im0, lambda),  
         lower = find_zero_triangle(re1, im1, lambda);
    
    float p, q;  
    if (lower) {
      p = lambda[0] + lambda[1]; 
      q = lambda[1]; 
    } else if (upper) {
      p = lambda[1]; 
      q = lambda[0] + lambda[1]; 
    } else {
      fprintf(stderr, "FATAL: punctured but no zero point.\n");
      p = q = 0.5f; 
      // return; 
    }

    float xx=x, yy=y, zz=z;  
    switch (face) {
    case FACE_XY: xx=x+p; yy=y+q; break; 
    case FACE_YZ: yy=y+p; zz=z+q; break; 
    case FACE_XZ: xx=x+p; zz=z+q; break;
    default: assert(0); break; 
    }
    
    // fprintf(stderr, "punctured, {%d, %d, %d}, face=%d, pt={%f, %f, %f}\n", 
    //     x, y, z, face, xx, yy, zz); 

    punctured_point_t point;
    point.x = xx; 
    point.y = yy; 
    point.z = zz; 
    point.flag = 0;
#else // bilinear
    float p[2] = {0.5, 0.5};
    if (!find_zero_quad(re, im, p)) {
      // fprintf(stderr, "FATAL: punctured but no zero point.\n");
      // return; 
    }

    float xx=x, yy=y, zz=z; 
    switch (face) {
    case FACE_XY: xx=x+p[0]; yy=y+p[1]; break; 
    case FACE_YZ: yy=y+p[0]; zz=z+p[1]; break; 
    case FACE_XZ: xx=x+p[0]; zz=z+p[1]; break; 
    default: assert(0); break; 
    }

    punctured_point_t point;
    point.x = xx; 
    point.y = yy; 
    point.z = zz; 
    point.flag = 0;

    // fprintf(stderr, "pos={%f, %f, %f}\n", xx, yy, zz);
#endif

    _points.insert(std::make_pair<int, punctured_point_t>(face2id(face, x, y, z), point)); 
  }
}

void GLGPUVortexExtractor::Extract()
{
  int dims[3] = {_ds->dims()[0], _ds->dims()[1], _ds->dims()[2]}; 
  float phase[4], delta[4];  

  for (int x=0; x<dims[0]-1; x++) 
    for (int y=0; y<dims[1]-1; y++) 
      for (int z=0; z<dims[2]-1; z++) 
        for (int face=0; face<3; face++) 
          solve(x, y, z, face);

  fprintf(stderr, "found %lu punctured points\n", _points.size());
}

void GLGPUVortexExtractor::Trace()
{
  // tracing punctured points
  while (!_points.empty()) {
    std::map<int, punctured_point_t>::iterator it = _points.begin();
    std::list<std::map<int, punctured_point_t>::iterator> traversed;

    trace(it, traversed, true, true); 
    trace(it, traversed, false, true);

    std::list<point_t> core;
    point_t pt;  

    // fprintf(stderr, "-----------\n"); 
    for (std::list<std::map<int, punctured_point_t>::iterator>::iterator it = traversed.begin(); it != traversed.end(); it++) {
      pt.x = (*it)->second.x; 
      pt.y = (*it)->second.y; 
      pt.z = (*it)->second.z;
      core.push_back(pt); 
      // fprintf(stderr, "{%f, %f, %f}\n", pt.x, pt.y, pt.z); 
      _points.erase(*it);
    }

    _cores.push_back(core); 
  }
  // fprintf(stderr, "traced %lu core lines\n", _cores.size()); 
}

void GLGPUVortexExtractor::trace(std::map<int, punctured_point_t>::iterator it, std::list<std::map<int, punctured_point_t>::iterator>& traversed, bool dir, bool seed)
{
  int f, x, y, z; 
  id2face(it->first, &f, &x, &y, &z); 
 
  // push_back/push_front
  if (!it->second.flag) {
    if (dir) traversed.push_back(it); 
    else traversed.push_front(it); 
  }
  it->second.flag = 1;
 
  // fprintf(stderr, "dir=%d, seed=%d, face={%d, %d, %d: %d}, pt={%f, %f, %f}\n",
  //     dir, seed, x, y, z, f, it->second.pt[0], it->second.pt[1], it->second.pt[2]); 

  std::map<int, punctured_point_t>::iterator next[10]; 
  switch (f) {
  case FACE_XY: 
    // upper
    next[0] = _points.find(face2id(FACE_YZ, x, y, z)); 
    next[1] = _points.find(face2id(FACE_XZ, x, y, z));
    next[2] = _points.find(face2id(FACE_YZ, x+1, y, z));
    next[3] = _points.find(face2id(FACE_XZ, x, y+1, z)); 
    next[4] = _points.find(face2id(FACE_XY, x, y, z+1)); 
    // lower
    next[5] = _points.find(face2id(FACE_YZ, x, y, z-1));
    next[6] = _points.find(face2id(FACE_XZ, x, y, z-1)); 
    next[7] = _points.find(face2id(FACE_XY, x, y, z-1));
    next[8] = _points.find(face2id(FACE_YZ, x+1, y, z-1));
    next[9] = _points.find(face2id(FACE_XZ, x, y+1, z-1));
    break;

  case FACE_YZ:
    // right
    next[0] = _points.find(face2id(FACE_XY, x, y, z));
    next[1] = _points.find(face2id(FACE_XZ, x, y, z)); 
    next[2] = _points.find(face2id(FACE_YZ, x+1, y, z)); 
    next[3] = _points.find(face2id(FACE_XZ, x, y+1, z));
    next[4] = _points.find(face2id(FACE_XY, x, y, z+1)); 
    // left
    next[5] = _points.find(face2id(FACE_XY, x-1, y, z)); 
    next[6] = _points.find(face2id(FACE_XZ, x-1, y, z)); 
    next[7] = _points.find(face2id(FACE_YZ, x-1, y, z));
    next[8] = _points.find(face2id(FACE_XZ, x-1, y+1, z)); 
    next[9] = _points.find(face2id(FACE_XY, x-1, y, z+1));
    break; 

  case FACE_XZ: 
    // front
    next[0] = _points.find(face2id(FACE_XY, x, y, z)); 
    next[1] = _points.find(face2id(FACE_YZ, x, y, z)); 
    next[2] = _points.find(face2id(FACE_YZ, x+1, y, z));
    next[3] = _points.find(face2id(FACE_XZ, x, y+1, z)); 
    next[4] = _points.find(face2id(FACE_XY, x, y, z+1));
    // back 
    next[5] = _points.find(face2id(FACE_XY, x, y-1, z)); 
    next[6] = _points.find(face2id(FACE_XZ, x, y-1, z)); 
    next[7] = _points.find(face2id(FACE_YZ, x, y-1, z));
    next[8] = _points.find(face2id(FACE_YZ, x+1, y-1, z)); 
    next[9] = _points.find(face2id(FACE_XY, x, y-1, z+1));
    break; 
  }

#if 0 
  int deg0 = 0, deg1 = 0; 
  for (int i=0; i<5; i++) 
    if (next[i] != _points.end()) 
      deg0 ++; 
  for (int i=0; i<5; i++) 
    if (next[i] != _points.end()) 
      deg1 ++; 
  if (deg0>1 || deg1>1)
    fprintf(stderr, "{%d, %d, %d, f=%d}, deg0=%d, deg1=%d\n", x, y, z, f, deg0, deg1);
#endif

  int start = 0, end = 10;
  if (seed && dir) end = 5;
  if (seed && !dir) start = 5;

  for (int i=start; i<end; i++) 
    if (next[i] != _points.end() && !next[i]->second.flag) {
      trace(next[i], traversed, dir, false);
      break; 
    }
}


////////////////////
void GLGPUVortexExtractor::id2cell(int id, int *x, int *y, int *z)
{
  int s = _ds->dims()[0] * _ds->dims()[1]; 
  *z = id / s; 
  *y = (id - *z*s) / _ds->dims()[0]; 
  *x = id - *z*s - *y*_ds->dims()[0]; 
}

int GLGPUVortexExtractor::cell2id(int x, int y, int z)
{
  return x + _ds->dims()[0] * (y + _ds->dims()[1] * z); 
}

int GLGPUVortexExtractor::face2id(int f, int x, int y, int z)
{
  if (f<0 || f>=3 || x<0 || y<0 || z<0 || x>=_ds->dims()[0] || y>=_ds->dims()[1] || z>=_ds->dims()[2]) return -1; // non-exist
  return f + 3*cell2id(x, y, z); 
}

void GLGPUVortexExtractor::id2face(int id, int *f, int *x, int *y, int *z)
{
  *f = id % 3; 
  id2cell(id/3, x, y, z); 
}

float GLGPUVortexExtractor::gauge(int *x0, int *x1) const
{
  float gx, gy, gz; 
  float dx[3] = {0.f};

#if 0
  for (int i=0; i<3; i++) {
    dx[i] = x1[i] - x0[i];
    if (dx[i] > _ds->dims()[i]*0.5f) 
      dx[i] -= _ds->dims()[i]*0.5f;
    else if (dx[i] < -_ds->dims()[i]*0.5f) 
      dx[i] += _ds->dims()[i]*0.5f; 
    dx[i] *= _ds->cellLengths()[i]; 
  }
#endif
  
  for (int i=0; i<3; i++) 
    dx[i] = (x1[i] - x0[i]) * _ds->cellLengths()[i];

  float x = ((x0[0] + x1[0])*0.5 - (_ds->dims()[0]-1)*0.5) * _ds->cellLengths()[0],  
        y = ((x0[1] + x1[1])*0.5 - (_ds->dims()[1]-1)*0.5) * _ds->cellLengths()[1], 
        z = ((x0[2] + x1[2])*0.5 - (_ds->dims()[2]-1)*0.5) * _ds->cellLengths()[2]; 

  if (_ds->B()[1] > 0) { // Y-Z gauge
    gx = dx[0] * _ds->Kex();
    gy =-dx[1] * x * _ds->Bz(); 
    gz = dx[2] * y * _ds->By(); 
  } else { // X-Z gauge
    gx = dx[0] * y * _ds->Bz()  //  dx*y^hat*Bz
        +dx[0] * _ds->Kex(); 
    gy = 0; 
    gz =-dx[2] * y * _ds->Bx(); // -dz*y^hat*Bx
  }

  return gx + gy + gz;  
}
