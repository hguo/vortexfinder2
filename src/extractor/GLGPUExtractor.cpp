#include <cmath>
#include <cassert>
#include "common/Utils.hpp"
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

void GLGPUVortexExtractor::SetInterpolationMode(int mode)
{
  _interpolation_mode = mode;
}

void GLGPUVortexExtractor::Extract()
{
  int idx[3], idx1[3];
  const GLGPUDataset *ds = (const GLGPUDataset*)_dataset; 

  for (int i=0; i<3; i++) 
    idx1[i] = ds->pbc()[i] ? ds->dims()[i] : ds->dims()[i]-1;

  for (idx[0]=0; idx[0]<idx1[0]; idx[0]++) 
    for (idx[1]=0; idx[1]<idx1[1]; idx[1]++) 
      for (idx[2]=0; idx[2]<idx1[2]; idx[2]++) {
        ElemIdType id = ds->Idx2ElemId(idx);
        ExtractElem(id);
      }
}

PuncturedElem* GLGPUVortexExtractor::NewPuncturedElem(ElemIdType id) const
{
  PuncturedElem *p = new PuncturedElemHex;
  p->Init();
  p->SetElemId(id);
  return p;
}
  
PuncturedElem* GLGPUVortexExtractor::NewPuncturedVirtualElem(FaceIdType id) const
{
  PuncturedElem *p = new PuncturedPrismQuad;
  p->Init();
  p->SetElemId(id);
  return p;
}
  
bool GLGPUVortexExtractor::FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const
{
  const double epsilon = 0.01;

  switch (_interpolation_mode) {
  case INTERPOLATION_CENTER: 
    return find_zero_quad_center(re, im, X, pos); 

  case INTERPOLATION_BARYCENTRIC: 
    return find_zero_quad_barycentric(re, im, X, pos, epsilon); 

  case INTERPOLATION_BILINEAR: 
    return find_zero_quad_bilinear(re, im, X, pos, epsilon); 
  
  case INTERPOLATION_LINECROSS: // TODO
    fprintf(stderr, "FATAL: line cross not yet implemented. exiting.\n"); 
    assert(false);
    return find_zero_quad_line_cross(re, im, X, pos, epsilon);

  default:
    return false;
  }
}
