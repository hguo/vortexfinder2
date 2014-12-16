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
  void Trace(const double seed[3]);

  bool RK1(double pt[3], double h);
  bool RK4(double pt[3], double h);
  
  bool Supercurrent(const double *X, double *J) const;

protected:
  const GLDataset *_ds;
  std::vector<FieldLine> _fieldlines;
}; 

#endif
