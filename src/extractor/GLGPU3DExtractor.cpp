#include <cmath>
#include <cassert>
#include "common/Utils.hpp"
#include "io/GLGPU3DDataset.h"
#include "GLGPU3DExtractor.h"
#include "InverseInterpolation.h"

GLGPU3DVortexExtractor::GLGPU3DVortexExtractor() :
  _interpolation_mode(INTERPOLATION_BILINEAR)
{
}

GLGPU3DVortexExtractor::~GLGPU3DVortexExtractor()
{
}

void GLGPU3DVortexExtractor::SetInterpolationMode(int mode)
{
  _interpolation_mode = mode;
}

void GLGPU3DVortexExtractor::Extract()
{
#if 0
  int idx[3], idx1[3];
  const GLGPU3DDataset *ds = (const GLGPU3DDataset*)_dataset; 

  for (int i=0; i<3; i++) 
    idx1[i] = ds->pbc()[i] ? ds->dims()[i] : ds->dims()[i]-1;

  for (idx[0]=0; idx[0]<idx1[0]; idx[0]++) 
    for (idx[1]=0; idx[1]<idx1[1]; idx[1]++) 
      for (idx[2]=0; idx[2]<idx1[2]; idx[2]++) {
        ElemIdType id = ds->Idx2ElemId(idx);
        ExtractElem(id);
      }
#endif
}

bool GLGPU3DVortexExtractor::FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const
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
