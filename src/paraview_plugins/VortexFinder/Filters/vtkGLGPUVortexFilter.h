#ifndef vtkGLGPUVortexFilter_h
#define vtkGLGPUVortexFilter_h

#include "vtkVortexFiltersModule.h"
#include "vtkImageAlgorithm.h"
#include "vtkPolyDataAlgorithm.h"

class vtkDataSet;

class VTKVORTEXFILTERS_EXPORT vtkGLGPUVortexFilter : public vtkImageAlgorithm
{
public:
  static vtkGLGPUVortexFilter *New();
  vtkTypeMacro(vtkGLGPUVortexFilter, vtkImageAlgorithm);

  void SetUseGPU(bool);
  void SetMeshType(int);
  void SetExtentThreshold(double);

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
  double dExtentThreshold;
};

#endif
