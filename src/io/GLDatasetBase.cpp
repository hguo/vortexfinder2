#include "GLDatasetBase.h"
#include "common/MeshGraph.h"

GLDatasetBase::GLDatasetBase() :
  _mg(NULL)
{
}

GLDatasetBase::~GLDatasetBase()
{
  if (_mg != NULL)
    delete _mg;
}

void GLDatasetBase::SerializeDataInfoToString(std::string& buf) const
{

}

void GLDatasetBase::SetDataName(const std::string& dn)
{
  _data_name = dn;
}

void GLDatasetBase::SetTimeStep(int t, int slot)
{
  _timestep[slot] = t;
}

void GLDatasetBase::RotateTimeSteps()
{
  int t = _timestep[1];
  _timestep[1] = _timestep[0];
  _timestep[0] = t;
}

int GLDatasetBase::TimeStep(int slot) const
{
  return _timestep[slot];
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

