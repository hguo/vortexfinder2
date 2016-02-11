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
  void SetPrecomputeSupercurrent(bool);
  
  virtual bool OpenDataFile(const std::string& filename); 
  virtual bool LoadTimeStep(int timestep, int slot) = 0;
  virtual void CloseDataFile();
  void RotateTimeSteps();

public: // mesh info
  virtual int Dimensions() const = 0;

  virtual void BuildMeshGraph() = 0;

public: // mesh utils
  virtual void GetFaceValues(const CFace&, int timeslot, float X[][3], float A[][3], float rho[], float phi[], float re[], float im[]) const;
  virtual void GetSpaceTimeEdgeValues(const CEdge&, float X[][3], float A[][3], float rho[], float phi[], float re[], float im[]) const;
  
  virtual CellIdType Pos2CellId(const float X[]) const = 0; //!< returns the elemId for a given position
  // virtual bool OnBoundary(ElemIdType id) const = 0;

public: // transformations and utils
  float LineIntegral(const float X0[], const float X1[], const float A0[], const float A1[]) const;
  virtual float QP(const float X0[], const float X1[], int slot=0) const;

public: // properties
  // Geometries
  const float* Origins() const {return _h[0].origins;}
  const float* Lengths() const {return _h[0].lengths;}

  // Time
  float Time(int slot=0) const {return _h[slot].time;}
  
  // Voltage
  float V(int slot=0) const {return _h[slot].V;}

  // Kx
  float Kex(int slot=0) const {return _h[slot].Kex;}
  float Kex_dot(int slot=0) const {return _h[slot].Kex_dot;}

  // Positions 
  virtual bool Pos(NodeIdType, float X[3]) const = 0;

  // Order parameters (direct access/linear interpolation)
  bool Rho(const float X[3], float &rho, int slot=0) const;
  bool Phi(const float X[3], float &phi, int slot=0) const;

  virtual float Rho(NodeIdType i, int slot=0) const = 0;
  virtual float Phi(NodeIdType i, int slot=0) const = 0;
  virtual float Re(NodeIdType i, int slot=0) const {return Rho(i, slot) * cos(Phi(i, slot));}
  virtual float Im(NodeIdType i, int slot=0) const {return Rho(i, slot) * sin(Phi(i, slot));}
  
  inline void RhoPhi(NodeIdType i, float &rho, float &phi, int slot=0) const {rho = Rho(i, slot); phi = Phi(i, slot);}
  inline void ReIm(NodeIdType i, float &re, float &im, int slot=0) const {re = Re(i, slot); im = Im(i, slot);}
  inline void RhoPhiReIm(NodeIdType i, float &rho, float &phi, float &re, float &im, int slot=0) const {rho = Rho(i, slot); phi = Phi(i, slot); re = Re(i, slot); im = Im(i, slot);}
  
  // Magnetic potential
  virtual bool A(const float X[3], float A[3], int slot=0) const = 0; //!< the vector potential at given position
  virtual bool A(NodeIdType, float A[3], int slot=0) const = 0;

  // Supercurrent field
  virtual bool Supercurrent(const float X[3], float J[3], int slot=0) const = 0;
  virtual bool Supercurrent(NodeIdType, float J[3], int slot) const = 0;

protected:
  std::vector<float> _time_stamps; 
  bool _valid;
  bool _precompute_supercurrent;
}; 

#endif
