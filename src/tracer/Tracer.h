#ifndef _TRACER_H
#define _TRACER_H

#include "common/FieldLine.h"

class GLDataset;

class FieldLineTracer {
public: 
  FieldLineTracer(); 
  ~FieldLineTracer(); 

  void SetDataset(const GLDataset* ds);
  void Trace();

  void WriteFieldLines(const std::string& filename);
 
protected:
  void Trace(const float seed[3]);

  template <typename T>
  bool RK1(T pt[3], T h);
  
  template <typename T>
  bool RK4(T pt[3], T h);
  
  bool Supercurrent(const float *X, float *J) const;

protected:
  const GLDataset *_ds;
  std::vector<FieldLine> _fieldlines;
}; 

#endif
