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

  virtual void PrintInfo() const = 0;
  virtual void SerializeDataInfoToString(std::string& buf) const = 0;

public: // data I/O
  virtual bool OpenDataFile(const std::string& filename); 
  virtual void LoadTimeStep(int timestep);
  virtual void CloseDataFile();

  virtual void ComputeSupercurrentField() = 0;

public: // mesh traversal & utils
  virtual std::vector<unsigned int> Neighbors(unsigned int elem_id) const = 0;

  double GaugeTransformation(const double X0[], const double X1[]) const;
  double Flux(const double X[3][3]) const; //!< flux for a triangle
  double Flux(int n, const double X[][3]) const; //!< flux for an arbitrary closed curve

  virtual unsigned int Pos2ElemId(const double X[]) const = 0; //!< returns the elemId for a given position

public: // properties
  int Dimensions() const {return 3;}  // currently only 3D data is supported

  int TimeStep() const {return _timestep;}

  // Magnetic field
  void SetMagneticField(const double B[3]);
  const double* B() const {return _B;}
  double Bx() const {return _B[0];} 
  double By() const {return _B[1];} 
  double Bz() const {return _B[2];}

  // Magnetic potential
  void A(const double X[3], double A[3]) const; //!< compute the vector potential at given position
  double Ax(const double X[3]) const {if (By()>0) return 0; else return -X[1]*Bz();}
  double Ay(const double X[3]) const {if (By()>0) return X[0]*Bz(); else return 0;}
  double Az(const double X[3]) const {if (By()>0) return -X[0]*By(); else return X[1]*Bx();}

  // Geometries
  const double* Origins() const {return _origins;}
  const double* Lengths() const {return _lengths;} 

  // Kx
  void SetKex(double Kex);
  double Kex() const {return _Kex;} 
  double Kex_dot() const {return _Kex_dot;}

  // Order parameters (direct access/linear interpolation)
  virtual bool Psi(const double X[3], double &re, double &im) const = 0;
  bool Rho(const double X[3], double &rho) const;
  bool Phi(const double X[3], double &phi) const;

  // Supercurrent field
  virtual bool Supercurrent(const double X[3], double J[3]) const = 0;

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
