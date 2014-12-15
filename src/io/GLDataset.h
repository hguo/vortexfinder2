#ifndef _GLDATASET_H
#define _GLDATASET_H

#include <string>
#include <vector>
#include "def.h"

class GLDataset 
{
public: 
  GLDataset(); 
  virtual ~GLDataset();

public: // data I/O
  virtual bool OpenDataFile(const std::string& filename); 
  virtual void LoadTimeStep(int timestep);
  virtual void CloseDataFile();

  virtual void SerializeDataInfoToString(std::string& buf) const = 0;

public: // mesh traversal & utils
  virtual std::vector<unsigned int> Neighbors(unsigned int elem_id) const = 0;

  double GaugeTransformation(const double *X0, const double *X1) const;
  double Flux(const double X[3][3]) const; //!< flux for a triangle
  double Flux(int n, const double **X) const; //!< flux for an arbitrary closed curve

public: // properties
  int Dimensions() const {return 3;}  // currently only 3D data is supported

  int TimeStep() const {return _timestep;}

  void SetMagneticField(const double B[3]);
  void SetKex(double Kex);

  const double* B() const {return _B;}
  double Bx() const {return _B[0];} 
  double By() const {return _B[1];} 
  double Bz() const {return _B[2];}

  void A(const double X[3], double A[3]) const; //!< compute the vector potential at given position
  double Ax(const double X[3]) const {return X[2]*By();}
  double Ay(const double X[3]) const {return X[0]*Bz();}
  double Az(const double X[3]) const {return X[1]*Bx();}

  const double* Origins() const {return _origins;}
  const double* Lengths() const {return _lengths;} 

  double Kex() const {return _Kex;} 
  double Kex_dot() const {return _Kex_dot;}

  virtual void PrintInfo() const = 0;

protected:
  int _timestep; 
  std::vector<double> _time_stamps; 

  std::string _data_name;

  double _origins[3]; 
  double _lengths[3];
  double _B[3];
  double _Kex, _Kex_dot;
  double _fluctuation_amp; 
}; 

#endif
