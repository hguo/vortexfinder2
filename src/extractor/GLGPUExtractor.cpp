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
  VortexExtractor::SetDataset(ds);
  _ds = (const GLGPUDataset*)ds; 
}

void GLGPUVortexExtractor::Extract()
{
  int idx[3]; 

  for (idx[0]=0; idx[0]<_ds->dims()[0]-1; idx[0]++) 
    for (idx[1]=0; idx[1]<_ds->dims()[1]-1; idx[1]++) 
      for (idx[2]=0; idx[2]<_ds->dims()[2]-1; idx[2]++) 
        ExtractElem(idx);
}

void GLGPUVortexExtractor::ExtractElem(int *idx)
{
  unsigned int elem_id = _ds->Idx2ElemId(idx);

  PuncturedElem *pelem = new PuncturedElemHex;
  pelem->Init();
  pelem->SetElemId(elem_id);

  for (int face=0; face<6; face++) {
    int X[4][3]; 
    double flux = _ds->Flux(face); 
    _ds->GetFace(idx, face, X);
    
    double re[4], im[4], amp[4], phase[4];
    double vertices[4][3];
    for (int i=0; i<4; i++) {
      _ds->Idx2Pos(X[i], vertices[i]);
      re[i] = _ds->re(X[i][0], X[i][1], X[i][2]); 
      im[i] = _ds->im(X[i][0], X[i][1], X[i][2]); 
      amp[i] = sqrt(re[i]*re[i] + im[i]*im[i]); 
      phase[i] = atan2(im[i], re[i]);
    }

    double delta[4];
    if (_gauge) {
      delta[0] = phase[1] - phase[0] + gauge(X[0], X[1]);  
      delta[1] = phase[2] - phase[1] + gauge(X[1], X[2]);  
      delta[2] = phase[3] - phase[2] + gauge(X[2], X[3]); 
      delta[3] = phase[0] - phase[3] + gauge(X[3], X[0]); 
    } else {
      delta[0] = phase[1] - phase[0];  
      delta[1] = phase[2] - phase[1];  
      delta[2] = phase[3] - phase[2];
      delta[3] = phase[0] - phase[3];
    }

    double sum = 0.f;
    double delta1[4];
    for (int i=0; i<4; i++) {
      delta1[i] = mod2pi(delta[i] + M_PI) - M_PI;
      sum += delta1[i]; 
    }
    sum += flux; 

    if (_gauge) {
      phase[1] = phase[0] + delta1[0]; 
      phase[2] = phase[1] + delta1[1]; 
      phase[3] = phase[2] + delta1[2];
      
      for (int i=0; i<4; i++) {
        re[i] = amp[i] * cos(phase[i]); 
        im[i] = amp[i] * sin(phase[i]); 
      }
    }

    double ps = sum / (2*M_PI);
    if (fabs(ps)<0.99f) 
      continue; 

    int chirality = ps>0 ? 1 : -1;
    double pos[3];
    // bool succ = find_zero_quad_centric(re, im, vertices, pos); 
    bool succ = find_zero_quad_bilinear(re, im, vertices, pos); 
    if (succ) {
      pelem->AddPuncturedFace(face, chirality, pos);
    } else {
      fprintf(stderr, "WARNING: punctured but singularities not found\n"); 
    }
  }

  if (pelem->Punctured()) {
    _punctured_elems[elem_id] = pelem;
  }
  else
    delete pelem;
}


#if 0 // find zero point
#if 0 // barycentric
    double re0[3] = {re[1], re[2], re[0]},
          im0[3] = {im[1], im[2], im[0]},
          re1[3] = {re[3], re[2], re[0]},
          im1[3] = {im[3], im[2], im[0]};
    double lambda[3];

    bool upper = find_zero_triangle(re0, im0, lambda),  
         lower = find_zero_triangle(re1, im1, lambda);
    
    double p, q;  
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

    double xx=x, yy=y, zz=z;  
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
    double p[2] = {0.5, 0.5};
    if (!find_zero_quad(re, im, p)) {
      // fprintf(stderr, "FATAL: punctured but no zero point.\n");
      // return; 
    }

    double xx=x, yy=y, zz=z; 
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
#endif



double GLGPUVortexExtractor::gauge(int *x0, int *x1) const
{
  double gx, gy, gz; 
  double dx[3] = {0.f};

#if 0
  for (int i=0; i<3; i++) {
    dx[i] = x1[i] - x0[i];
    if (dx[i] > _ds->dims()[i]*0.5f) 
      dx[i] -= _ds->dims()[i]*0.5f;
    else if (dx[i] < -_ds->dims()[i]*0.5f) 
      dx[i] += _ds->dims()[i]*0.5f; 
    dx[i] *= _ds->CellLengths()[i]; 
  }
#endif
  
  for (int i=0; i<3; i++) 
    dx[i] = (x1[i] - x0[i]) * _ds->CellLengths()[i];

  double x = ((x0[0] + x1[0])*0.5 - (_ds->dims()[0]-1)*0.5) * _ds->CellLengths()[0],  
         y = ((x0[1] + x1[1])*0.5 - (_ds->dims()[1]-1)*0.5) * _ds->CellLengths()[1], 
         z = ((x0[2] + x1[2])*0.5 - (_ds->dims()[2]-1)*0.5) * _ds->CellLengths()[2]; 

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
