#ifndef __vtkGLGPUVortexFilter_h
#define __vtkGLGPUVortexFilter_h

#include "vtkImageAlgorithm.h"
#include "vtkPolyDataAlgorithm.h"

class vtkDataSet;

class vtkGLGPUVortexFilter : public vtkImageAlgorithm
{
public:
  static vtkGLGPUVortexFilter *New();
  vtkTypeRevisionMacro(vtkGLGPUVortexFilter, vtkImageAlgorithm);

protected:
  vtkGLGPUVortexFilter();
  ~vtkGLGPUVortexFilter();

  virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  
  int FillOutputPortInformation(int, vtkInformation*);

private:
  vtkGLGPUVortexFilter(const vtkGLGPUVortexFilter&);
  void operator=(const vtkGLGPUVortexFilter&);
};

#endif
