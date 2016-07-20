#ifndef _GLGPUDATASET_H
#define _GLGPUDATASET_H

#include "io/GLDataset.h"

class GLGPUDataset : public GLDataset
{
public:
  GLGPUDataset();
  ~GLGPUDataset();
  
  void SerializeDataInfoToString(std::string& buf) const;
  
public:
  bool OpenDataFile(const std::string& filename); // file list
  bool OpenDataFileByPattern(const std::string& pattern); 
  bool LoadTimeStep(int timestep, int slot=0);
  void RotateTimeSteps();
  void CloseDataFile();

  int NTimeSteps() const {return _filenames.size();}

  void PrintInfo(int slot=0) const;

  bool BuildDataFromArray(const GLHeader&, const float *rho, const float *phi, const float *re, const float *im);
  void GetDataArray(GLHeader& h, float **rho, float **phi, float **re, float **im, float **J, int slot=0);
  float *GetSupercurrentDataArray() const {return _J[0];} // FIXME
  
private:
  bool OpenBDATDataFile(const std::string& filename, int slot=0);
  bool OpenLegacyDataFile(const std::string& filename, int slot=0);

  // void ComputeSupercurrentField(int slot=0);

protected:
  void Nid2Idx(NodeIdType id, int *idx) const; 
  NodeIdType Idx2Nid(const int *idx) const;
 
  void Idx2Pos(const int idx[3], float X[3]) const;
  void Pos2Idx(const float X[3], int idx[3]) const;
  void Pos2Grid(const float pos[3], float gpos[3]) const; //!< to grid coordinates

public: // rectilinear grid
  const int* dims() const {return _h[0].dims;}
  const bool* pbc() const {return _h[0].pbc;}
  const float* CellLengths() const {return _h[0].cell_lengths;}

  float dx() const {return _h[0].cell_lengths[0];}
  float dy() const {return _h[0].cell_lengths[1];}
  float dz() const {return _h[0].cell_lengths[2];}
  
  // Magnetic potential
  bool A(const float X[3], float A[3], int slot=0) const;
  bool A(NodeIdType n, float A[3], int slot=0) const;
  
  // Magnetic field
  const float* B(int slot=0) const {return _h[slot].B;}

  // bool Psi(NodeIdType, float &rho, float &phi, int slot=0) const;
  // bool Psi(const float X[3], float &rho, float &phi, int slot=0) const;

  inline float Rho(NodeIdType i, int slot=0) const {return _rho[slot][i];}
  inline float Phi(NodeIdType i, int slot=0) const {return _phi[slot][i];}
  inline float Re(NodeIdType i, int slot=0) const {return _re[slot][i];}
  inline float Im(NodeIdType i, int slot=0) const {return _im[slot][i];}

  float Rho(int i, int j, int k, int slot=0) const; 
  float Phi(int i, int j, int k, int slot=0) const;
  float Re(int i, int j, int k, int slot=0) const; 
  float Im(int i, int j, int k, int slot=0) const;
  
  bool Pos(NodeIdType, float X[3]) const;
  bool Supercurrent(const float X[3], float J[3], int slot=0) const;
  bool Supercurrent(NodeIdType, float J[3], int slot=0) const;

public:
  float QP(const float X0[], const float X1[], int slot=0) const;

protected:
  float *_rho[2], *_phi[2], *_re[2], *_im[2];
  float *_J[2]; // supercurrent

  std::vector<std::string> _filenames; // filenames for different timesteps
};

#endif
