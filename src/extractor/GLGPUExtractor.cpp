#include <cmath>
#include <cassert>
#include "io/GLGPUDataset.h"
#include "GLGPUExtractor.h"
#include "InverseInterpolation.h"

GLGPUVortexExtractor::GLGPUVortexExtractor() :
  _interpolation_mode(INTERPOLATION_BILINEAR)
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

void GLGPUVortexExtractor::SetInterpolationMode(int mode)
{
  _interpolation_mode = mode;
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
      delta[0] = phase[1] - phase[0] + _ds->GaugeTransformation(vertices[0], vertices[1]);  
      delta[1] = phase[2] - phase[1] + _ds->GaugeTransformation(vertices[1], vertices[2]);  
      delta[2] = phase[3] - phase[2] + _ds->GaugeTransformation(vertices[2], vertices[3]); 
      delta[3] = phase[0] - phase[3] + _ds->GaugeTransformation(vertices[3], vertices[0]); 
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
    bool succ = false; 

    switch (_interpolation_mode) {
    case INTERPOLATION_CENTER: 
      succ = find_zero_quad_center(re, im, vertices, pos); 
      break;

    case INTERPOLATION_BARYCENTRIC: 
      succ = find_zero_quad_barycentric(re, im, vertices, pos); 
      break; 

    case INTERPOLATION_BILINEAR: 
      succ = find_zero_quad_bilinear(re, im, vertices, pos); 
      break;
    
    case INTERPOLATION_LINECROSS: // TODO
      fprintf(stderr, "FATAL: line cross not yet implemented. exiting.\n"); 
      assert(false);
      succ = find_zero_quad_line_cross(re, im, vertices, pos); 
      break;

    default: 
      break;
    }
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
