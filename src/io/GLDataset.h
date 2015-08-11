#ifndef _GLDATASET_H
#define _GLDATASET_H

#include "GLDatasetBase.h"
#include <cmath>

class GLDataset : public GLDatasetBase
{
public: 
  GLDataset(); 
  virtual ~GLDataset();

  bool Valid() const {return _valid;}
  virtual void PrintInfo(int slot=0) const = 0;

public: // data I/O
  virtual bool OpenDataFile(const std::string& filename); 
  virtual void LoadTimeStep(int timestep, int slot) = 0;
  virtual void CloseDataFile();
  void RotateTimeSteps();

public: // mesh info
  virtual int Dimensions() const = 0;
  virtual int NrFacesPerCell() const = 0;
  virtual int NrNodesPerFace() const = 0;

  virtual void BuildMeshGraph() = 0;

public: // mesh utils
  virtual void GetFaceValues(const CFace&, int timeslot, double X[][3], double A[][3], double rho[], double phi[]) const;
  virtual void GetSpaceTimeEdgeValues(const CEdge&, double X[][3], double A[][3], double rho[], double phi[]) const;
  
  virtual CellIdType Pos2CellId(const double X[]) const = 0; //!< returns the elemId for a given position
  // virtual bool OnBoundary(ElemIdType id) const = 0;

public: // transformations and utils
  double LineIntegral(const double X0[], const double X1[], const double A0[], const double A1[]) const;
  virtual double QP(const double X0[], const double X1[], int slot=0) const;

public: // properties
  // Geometries
  const double* Origins() const {return _h[0].origins;}
  const double* Lengths() const {return _h[0].lengths;}

  // Time
  double Time(int slot=0) const {return _h[slot].time;}
  
  // Voltage
  double V(int slot=0) const {return _h[slot].V;}

  // Kx
  double Kex(int slot=0) const {return _h[slot].Kex;}
  double Kex_dot(int slot=0) const {return _h[slot].Kex_dot;}

  // Positions 
  virtual bool Pos(NodeIdType, double X[3]) const = 0;

  // Order parameters (direct access/linear interpolation)
  bool Rho(const double X[3], double &rho, int slot=0) const;
  bool Phi(const double X[3], double &phi, int slot=0) const;

  virtual double Rho(NodeIdType i, int slot=0) const = 0;
  virtual double Phi(NodeIdType i, int slot=0) const = 0;
  inline double Re(NodeIdType i, int slot=0) const {return Rho(i, slot) * cos(Phi(i, slot));}
  inline double Im(NodeIdType i, int slot=0) const {return Rho(i, slot) * sin(Phi(i, slot));}
  
  inline void RhoPhi(NodeIdType i, double &rho, double &phi, int slot=0) const {rho = Rho(i, slot); phi = Phi(i, slot);}
  inline void ReIm(NodeIdType i, double &re, double &im, int slot=0) const {re = Re(i, slot); im = Im(i, slot);}
  
  // Magnetic potential
  virtual bool A(const double X[3], double A[3], int slot=0) const = 0; //!< the vector potential at given position
  virtual bool A(NodeIdType, double A[3], int slot=0) const = 0;

  // Supercurrent field
  virtual bool Supercurrent(const double X[3], double J[3], int slot=0) const = 0;
  virtual bool Supercurrent(NodeIdType, double J[3], int slot) const = 0;

protected:
  std::vector<double> _time_stamps; 
  bool _valid;
}; 

#endif
