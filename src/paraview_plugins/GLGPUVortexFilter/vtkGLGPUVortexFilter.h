#ifndef __vtkGLGPUVortexFilter_h
#define __vtkGLGPUVortexFilter_h

#include "vtkImageAlgorithm.h"
#include "vtkPolyDataAlgorithm.h"

class vtkDataSet;

class vtkGLGPUVortexFilter : public vtkImageAlgorithm
{
public:
  static vtkGLGPUVortexFilter *New();
  vtkTypeMacro(vtkGLGPUVortexFilter, vtkImageAlgorithm);

  void SetUseGPU(bool);
  void SetMeshType(int);
  void SetLoopThreshold(double);

protected:
  vtkGLGPUVortexFilter();
  ~vtkGLGPUVortexFilter();

  virtual int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  
  int FillOutputPortInformation(int, vtkInformation*);

private:
  int ExtractVorticies(vtkImageData*, vtkPolyData*);

private:
  vtkGLGPUVortexFilter(const vtkGLGPUVortexFilter&);
  void operator=(const vtkGLGPUVortexFilter&);

private:
  bool bUseGPU;
  int iMeshType;
  double dLoopThreshold;
};

#endif
