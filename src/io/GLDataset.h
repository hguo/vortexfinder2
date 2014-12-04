#ifndef _GLDATASET_H
#define _GLDATASET_H

#include <string>
#include <vector>

class CGLDataset 
{
public: 
  CGLDataset(); 
  ~CGLDataset();

  int Dimensions() const {return 3;}  // currently only 3D data is supported

  const double* B() const {return _B;}
  double Bx() const {return _B[0];} 
  double By() const {return _B[1];} 
  double Bz() const {return _B[2];}

  const double* Lengths() const {return _Lengths;} 

  double Jx() const {return _Jx;}
  double Kex() const {return _Kx;} 
  double Kex_dot() const {return _Kex_dot;}

private:
  std::vector<double> _time_stamps; 
  double _Lengths[3];
  double _B[3];
  double _Jx; 
  double _Kex, _Kex_dot; 
}; 

#endif
