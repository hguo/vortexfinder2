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

  const int nseeds[3] = {8, 8, 8};
  const double span[3] = {
    _ds->Lengths()[0]/(nseeds[0]-1), 
    _ds->Lengths()[1]/(nseeds[1]-1), 
    _ds->Lengths()[2]/(nseeds[2]-1)}; 

  for (int i=0; i<nseeds[0]; i++) {
    for (int j=0; j<nseeds[1]; j++) {
      for (int k=0; k<nseeds[2]; k++) {
        double seed[3] = {
          i * span[0] + _ds->Origins()[0], 
          j * span[1] + _ds->Origins()[1], 
          k * span[2] + _ds->Origins()[2]}; 
        Trace(seed);
      }
    }
  }
}

void GLGPUFieldLineTracer::Trace(const double seed[3])
{
  static const int max_length = 1024; 
  const double h = 0.5 * std::min(_ds->CellLengths()[0], std::min(_ds->CellLengths()[1], _ds->CellLengths()[2]));
  double pt[3] = {seed[0], seed[1], seed[2]}; 
  int n = 0;

  std::list<double> line;

  // forward
  while (1) {
    line.push_back(pt[0]); line.push_back(pt[1]); line.push_back(pt[2]); 
    if (!RK1(pt, h)) break;

    n++;
    if (n>max_length-1) break; 
  }
 
  // backward
  n = 0;
  pt[0] = seed[0]; pt[1] = seed[1]; pt[2] = seed[2];
  line.pop_front(); line.pop_front(); line.pop_front(); 
  while (1) {
    line.push_front(pt[2]); line.push_front(pt[1]); line.push_front(pt[0]); 
    if (!RK1(pt, -h)) break;

    n++;
    if (n>max_length-1) break; 
  }

  FieldLine line1(line);
  _fieldlines.push_back(line1);
}

bool GLGPUFieldLineTracer::RK1(double *pt, double h)
{
  double v[3], gpt[3]; 
  const int st[3] = {0}, 
            sz[3] = {_ds->dims()[0], _ds->dims()[1], _ds->dims()[2]}; 

  _ds->Pos2Grid(pt, gpt);
  if (gpt[0]<1 || gpt[0]>sz[0]-2 || 
      gpt[1]<1 || gpt[1]>sz[1]-2 || 
      gpt[2]<1 || gpt[2]>sz[2]-2) return false;

  if (!lerp3D(gpt, st, sz, 3, _sc, v)) return false; 

  pt[0] = pt[0] + h*v[0]; 
  pt[1] = pt[1] + h*v[1]; 
  pt[2] = pt[2] + h*v[2];

  return true; 
}
