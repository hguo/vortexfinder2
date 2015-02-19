#ifndef _GLGPUDATASET_H
#define _GLGPUDATASET_H

#include "io/GLDataset.h"

class GLGPUDataset : public GLDataset
{
public:
  GLGPUDataset();
  ~GLGPUDataset();
  
public:
  bool OpenDataFile(const std::string& filename); 
  bool OpenBDATDataFile(const std::string& filename);
  bool OpenLegacyDataFile(const std::string& filename);

protected:
  int _dims[3]; 
  bool _pbc[3]; 
  double _cell_lengths[3]; 

  double _B[3]; // magnetic field

  double *_re, *_im, 
         *_re1, *_im1;
  double *_Jx, *_Jy, *_Jz; // only for timestep 0
};

#endif
