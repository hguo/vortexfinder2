#include "vtkGLGPUVortexFilter.h"
#include "vtkInformation.h"
#include "vtkSmartPointer.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkImageData.h"
#include "vtkSphereSource.h"
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

int vtkGLGPUVortexFilter::FillOutputPortInformation(int, vtkInformation *info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
  return 1;
}

int vtkGLGPUVortexFilter::RequestData(
    vtkInformation*, 
    vtkInformationVector** inputVector, 
    vtkInformationVector* outputVector)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  // TODO: check compatability
  vtkSmartPointer<vtkDataArray> dataArrayRe, dataArrayIm;
  int index;

  dataArrayRe = input->GetPointData()->GetArray("re", index);
  dataArrayIm = input->GetPointData()->GetArray("im", index);

  // dummy output
  vtkSmartPointer<vtkSphereSource> source = vtkSphereSource::New();
  source->SetRadius(30.0);
  source->SetCenter(0, 0, 0);
  source->Update();

  output->DeepCopy(source->GetOutput());

  return 1;
}
