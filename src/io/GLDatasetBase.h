#ifndef _GLDATASET_BASE_H
#define _GLDATASET_BASE_H

#include <string>
#include <vector>
#include "def.h"
#include "common/MeshGraph.h"
#include "GLHeader.h"

class GLDatasetBase
{
public:
  GLDatasetBase(); 
  virtual ~GLDatasetBase();
  
  virtual void SerializeDataInfoToString(std::string& buf) const;

public:
  void SetDataName(const std::string& dn);
  std::string DataName() const {return _data_name;}

  void SetTimeStep(int timestep, int slot=0);
  int TimeStep(int slot=0) const;
  virtual int NTimeSteps() const {return 0;}
  virtual void RotateTimeSteps();

  bool LoadMeshGraph(const std::string& filename);
  bool LoadDefaultMeshGraph();
  void SaveMeshGraph(const std::string& filename);
  void SaveDefaultMeshGraph();

  const struct MeshGraph* MeshGraph() const {return _mg;}

protected: 
  struct MeshGraph *_mg;
  std::string _data_name;
  GLHeader _h[2];
  int _timestep[2];
};

#endif
