#ifndef __vtkGLGPUSupercurrentFilter_h
#define __vtkGLGPUSupercurrentFilter_h

#include "vtkImageAlgorithm.h"

class vtkDataSet;

class vtkGLGPUSupercurrentFilter : public vtkImageAlgorithm
{
public:
  static vtkGLGPUSupercurrentFilter *New();
  vtkTypeMacro(vtkGLGPUSupercurrentFilter, vtkImageAlgorithm);

protected:
  vtkGLGPUSupercurrentFilter();
  ~vtkGLGPUSupercurrentFilter();

  virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  
  int FillOutputPortInformation(int, vtkInformation *info);

private:
  int ComputeSupercurrent(vtkImageData*, vtkImageData*);

private:
  vtkGLGPUSupercurrentFilter(const vtkGLGPUSupercurrentFilter&);
  void operator=(const vtkGLGPUSupercurrentFilter&);
};

#endif
