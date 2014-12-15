#include "GLGPUTracer.h"
#include "io/GLGPUDataset.h"
#include "common/Lerp.hpp"

GLGPUFieldLineTracer::GLGPUFieldLineTracer()
{
  memset(_sc, 0, sizeof(double*)*3);
}

GLGPUFieldLineTracer::~GLGPUFieldLineTracer()
{
}

void GLGPUFieldLineTracer::SetDataset(const GLDataset *ds)
{
  _ds = (const GLGPUDataset*)ds; 
  _ds->GetSupercurrentField(_sc);
}

void GLGPUFieldLineTracer::Trace()
{
  fprintf(stderr, "Trace..\n");
  
  double seed[3] = {-5, 0, 0};
  Trace(seed);
}

void GLGPUFieldLineTracer::Trace(const double seed[3])
{
  static const int max_length = 1024; 
  const double h = 0.5 * std::min(_ds->CellLengths()[0], std::min(_ds->CellLengths()[1], _ds->CellLengths()[2]));
  double pt[3] = {seed[0], seed[1], seed[2]}; 
  int n = 0;

  while (1) {
    fprintf(stderr, "{%f, %f, %f}\n", pt[0], pt[1], pt[2]);

    bool succ = rk1(pt, h);
    if (!succ) break;

    n++;
    if (n>max_length-1) break; 
  }
}

bool GLGPUFieldLineTracer::rk1(double *pt, double h)
{
  double v[3], gpt[3]; 
  const int st[3] = {0}, 
            sz[3] = {_ds->dims()[0], _ds->dims()[1], _ds->dims()[2]}; 

  _ds->Pos2Grid(pt, gpt);
  if (!lerp3D(gpt, st, sz, 3, _sc, v)) return false; 

  pt[0] = pt[0] + h*v[0]; 
  pt[1] = pt[1] + h*v[1]; 
  pt[2] = pt[2] + h*v[2];

  return true; 
}
