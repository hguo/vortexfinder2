#include "vtkGLGPUVortexFilter.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

vtkStandardNewMacro(vtkGLGPUVortexFilter);

vtkGLGPUVortexFilter::vtkGLGPUVortexFilter()
{
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

vtkGLGPUVortexFilter::~vtkGLGPUVortexFilter()
{
}
  
int vtkGLGPUVortexFilter::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*)
{
  return 1;
}

int vtkGLGPUVortexFilter::RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*)
{
  return 1;
}
