#ifndef _TRACER_H
#define _TRACER_H

#include "io/GLDataset.h"
#include "vortex/FieldLine.h"

class FieldLineTracer {
public: 
  FieldLineTracer(); 
  ~FieldLineTracer(); 

  virtual void SetDataset(const GLDataset* ds) = 0;
}; 

#endif
