#ifndef _GLGPUTRACER_H
#define _GLGPUTRACER_H

#include "Tracer.h"

class GLGPUDataset;

class GLGPUFieldLineTracer : public FieldLineTracer
{
public:
  GLGPUFieldLineTracer(); 
  ~GLGPUFieldLineTracer();

  void SetDataset(const GLDataset *ds);

  void Trace(); 

protected:
  void Trace(const double seed[3]);
 
protected:
  bool RK1(double pt[3], double h);
  bool RK4(double pt[3], double h);

private:
  const GLGPUDataset *_ds;
  const double *_sc[3];
}; 

#endif
