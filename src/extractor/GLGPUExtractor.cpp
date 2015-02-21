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

bool GLGPUVortexExtractor::FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const
{
  const double epsilon = 0.05;
  bool succ = false;

  switch (_interpolation_mode) {
  case INTERPOLATION_CENTER: 
    succ = find_zero_quad_center(re, im, X, pos);
    break;

  case INTERPOLATION_BARYCENTRIC: 
    succ = find_zero_quad_barycentric(re, im, X, pos, epsilon); 
    break;

  case INTERPOLATION_BILINEAR: 
    succ = find_zero_quad_bilinear(re, im, X, pos, epsilon);
    if (!succ)
      succ = find_zero_quad_barycentric(re, im, X, pos, epsilon); 
    break;
  
  case INTERPOLATION_LINECROSS: // TODO
    fprintf(stderr, "FATAL: line cross not yet implemented. exiting.\n"); 
    assert(false);
    break;
    // return find_zero_quad_line_cross(re, im, X, pos, epsilon);

  default:
    return false;
  }
 
  return succ;
}
