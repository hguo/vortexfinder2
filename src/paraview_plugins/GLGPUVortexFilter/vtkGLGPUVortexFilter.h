#ifndef __vtkGLGPUVortexFilter_h
#define __vtkGLGPUVortexFilter_h

#include "vtkPolyDataAlgorithm.h"

class vtkDataSet;

class vtkGLGPUVortexFilter : public vtkPolyDataAlgorithm
{
public:
  static vtkGLGPUVortexFilter *New();
  vtkTypeRevisionMacro(vtkGLGPUVortexFilter, vtkPolyDataAlgorithm);

  // virtual int ProcessRequest(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

protected:
  vtkGLGPUVortexFilter();
  ~vtkGLGPUVortexFilter();

  // virtual int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

private:
  vtkGLGPUVortexFilter(const vtkGLGPUVortexFilter&);
  void operator=(const vtkGLGPUVortexFilter&);
};

#endif
