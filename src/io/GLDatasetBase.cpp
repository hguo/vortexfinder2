#include "GLDatasetBase.h"

GLDatasetBase::GLDatasetBase() :
  _timestep(-1), _timestep1(-1), 
  _mg(NULL)
{
}

GLDatasetBase::~GLDatasetBase()
{
  if (_mg != NULL)
    delete _mg;
}

void GLDatasetBase::SetDataName(const std::string& dn)
{
  _data_name = dn;
}

void GLDatasetBase::SetTimeStep(int t, int slot)
{
  if (slot == 0)
    _timestep = t;
  else { // rotate
    if (_timestep1 != -1) 
      _timestep = _timestep1;
    _timestep1 = t;
  }
}

int GLDatasetBase::TimeStep(int slot) const
{
  return slot == 0 ? _timestep : _timestep1;
}

bool GLDatasetBase::LoadDefaultMeshGraph()
{
  if (_mg == NULL)
    _mg = new class MeshGraph;
  return LoadMeshGraph(_data_name + ".mg");
}

bool GLDatasetBase::LoadMeshGraph(const std::string& filename)
{
  return _mg->ParseFromFile(filename);
}

void GLDatasetBase::SaveDefaultMeshGraph()
{
  SaveMeshGraph(_data_name + ".mg");
}

void GLDatasetBase::SaveMeshGraph(const std::string& filename)
{
  _mg->SerializeToFile(filename);
}

