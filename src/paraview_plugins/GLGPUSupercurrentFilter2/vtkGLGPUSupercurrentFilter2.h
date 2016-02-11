#ifndef __vtkGLGPUSupercurrentFilter2_h
#define __vtkGLGPUSupercurrentFilter2_h

#include "vtkImageAlgorithm.h"

class vtkDataSet;

class vtkGLGPUSupercurrentFilter2 : public vtkImageAlgorithm
{
public:
  static vtkGLGPUSupercurrentFilter2 *New();
  vtkTypeMacro(vtkGLGPUSupercurrentFilter2, vtkImageAlgorithm);

protected:
  vtkGLGPUSupercurrentFilter2();
  ~vtkGLGPUSupercurrentFilter2();

  int RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

private:
  int ComputeSupercurrent(vtkImageData*, vtkImageData*);

private:
  vtkGLGPUSupercurrentFilter2(const vtkGLGPUSupercurrentFilter2&);
  void operator=(const vtkGLGPUSupercurrentFilter2&);
};

#endif
