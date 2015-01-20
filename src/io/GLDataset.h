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

  bool Valid() const {return _valid;}
  virtual void PrintInfo() const = 0;
  virtual void SerializeDataInfoToString(std::string& buf) const = 0;

public: // data I/O
  virtual bool OpenDataFile(const std::string& filename); 
  virtual void LoadTimeStep(int timestep);
  virtual void CloseDataFile();

public: // mesh info
  virtual int Dimensions() const = 0;
  virtual int NrFacesPerElem() const = 0;
  virtual int NrNodesPerFace() const = 0;

public: // mesh utils
  virtual std::vector<ElemIdType> GetNeighborIds(ElemIdType elem_id) const = 0;
  virtual bool GetFace(ElemIdType id, int face, double X[][3], double A[][3], double re[], double im[]) const = 0;
  virtual ElemIdType Pos2ElemId(const double X[]) const = 0; //!< returns the elemId for a given position
  virtual bool OnBoundary(ElemIdType id) const = 0;
  
  virtual bool GetSpaceTimePrism(ElemIdType id, int face, double X[][3], 
      double A0[][3], double A1[][3], 
      double re0[], double re1[],
      double im0[], double im1[]) const;

public: // transformations and utils
  virtual double GaugeTransformation(const double X0[], const double X1[], const double A0[], const double A1[]) const;
  double LineIntegral(const double X0[], const double X1[], const double A0[], const double A1[]) const;

  // double Flux(const double X[3][3]) const; //!< flux for a triangle
  // double Flux(int n, const double X[][3]) const; //!< flux for an arbitrary closed curve
  virtual double QP(const double X0[], const double X1[]) const;

public: // properties
  int TimeStep() const {return _timestep;}

  // Voltage
  double V() const {return _V;}

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
  
  // Magnetic potential
  virtual bool A(const double X[3], double A[3]) const = 0; //!< the vector potential at given position

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
  double _Jx;
  double _V;
  double _fluctuation_amp; 

  bool _valid;
}; 

#endif
