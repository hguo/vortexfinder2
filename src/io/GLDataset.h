#ifndef _GLDATASET_H
#define _GLDATASET_H

#include <string>
#include <vector>
#include "def.h"

class GLDataset 
{
public: 
  GLDataset(); 
  ~GLDataset();

  int Dimensions() const {return 3;}  // currently only 3D data is supported

  const double* B() const {return _B;}
  double Bx() const {return _B[0];} 
  double By() const {return _B[1];} 
  double Bz() const {return _B[2];}

  const double* Lengths() const {return _lengths;} 

  double Jx() const {return _Jx;}
  double Kex() const {return _Kex;} 
  double Kex_dot() const {return _Kex_dot;}

  virtual void PrintInfo() const = 0;

protected:
  std::vector<double> _time_stamps; 
  double _lengths[3];
  double _B[3];
  double _Jx; 
  double _Kex, _Kex_dot;
  double _fluctuation_amp; 
}; 

#endif
