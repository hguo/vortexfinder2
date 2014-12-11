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

private:
  const GLGPUDataset *_ds;
}; 

#endif
