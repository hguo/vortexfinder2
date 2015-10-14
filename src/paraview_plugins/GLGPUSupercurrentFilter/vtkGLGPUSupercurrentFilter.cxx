#include "vtkGLGPUSupercurrentFilter.h"
#include "vtkInformation.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "io/GLGPU3DDataset.h"

vtkStandardNewMacro(vtkGLGPUSupercurrentFilter);

vtkGLGPUSupercurrentFilter::vtkGLGPUSupercurrentFilter()
{
  SetNumberOfInputPorts(1);
  SetNumberOfOutputPorts(1);
}

vtkGLGPUSupercurrentFilter::~vtkGLGPUSupercurrentFilter()
{
}

int vtkGLGPUSupercurrentFilter::FillOutputPortInformation(int, vtkInformation *info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
  return 1;
}

int vtkGLGPUSupercurrentFilter::RequestData(
    vtkInformation*, 
    vtkInformationVector** inputVector, 
    vtkInformationVector* outputVector)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData *output = vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  return ComputeSupercurrent(input, output);
}

int vtkGLGPUSupercurrentFilter::ComputeSupercurrent(vtkImageData* inputData, vtkImageData* outputData)
{
  // TODO: check compatability
  vtkSmartPointer<vtkDataArray> dataArrayRho, dataArrayPhi, dataArrayRe, dataArrayIm;
  vtkSmartPointer<vtkDataArray> dataArrayB, dataArrayPBC, dataArrayJxext, dataArrayKx, dataArrayV;
  int index;

  dataArrayRho = inputData->GetPointData()->GetArray("rho", index);
  dataArrayPhi = inputData->GetPointData()->GetArray("phi", index);
  dataArrayRe = inputData->GetPointData()->GetArray("re", index);
  dataArrayIm = inputData->GetPointData()->GetArray("im", index);
  dataArrayB = inputData->GetFieldData()->GetArray("B", index);
  dataArrayPBC = inputData->GetFieldData()->GetArray("pbc", index);
  dataArrayJxext = inputData->GetFieldData()->GetArray("Jxext", index);
  dataArrayKx = inputData->GetFieldData()->GetArray("Kx", index);
  dataArrayV = inputData->GetFieldData()->GetArray("V", index);

  GLHeader h;
  h.ndims = 3;
  inputData->GetDimensions(h.dims);
  inputData->GetOrigin(h.origins);
  inputData->GetSpacing(h.cell_lengths);
  for (int i=0; i<3; i++) 
    h.lengths[i] = h.cell_lengths[i] * h.dims[i];

  dataArrayB->GetTuple(0, h.B);
  
  double pbc1[3];
  dataArrayPBC->GetTuple(0, pbc1);
  for (int i=0; i<3; i++)
    // h.pbc[i] = (pbc1[i]>0);
    h.pbc[i] = 0; 

  h.Jxext = dataArrayJxext->GetTuple1(0);
  h.Kex = dataArrayKx->GetTuple1(0);
  h.V = dataArrayV->GetTuple1(0);

  // fprintf(stderr, "B={%f, %f, %f}, pbc={%d, %d, %d}, Jxext=%f, Kx=%f, V=%f\n", 
  //     B[0], B[1], B[2], pbc[0], pbc[1], pbc[2], Jxext, Kx, V);

  const int arraySize = h.dims[0]*h.dims[1]*h.dims[2];
  double *rho = (double*)dataArrayRho->GetVoidPointer(0), 
         *phi = (double*)dataArrayPhi->GetVoidPointer(0), 
         *re = (double*)dataArrayRe->GetVoidPointer(0), 
         *im = (double*)dataArrayIm->GetVoidPointer(0);

  // build data
  GLGPU3DDataset *ds = new GLGPU3DDataset;
  ds->BuildDataFromArray(h, rho, phi, re, im);
  ds->ComputeSupercurrentField();
  const double *J = ds->GetSupercurrentDataArray();

  vtkSmartPointer<vtkDataArray> dataArrayJ; 
  dataArrayJ.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayJ->SetNumberOfComponents(3); 
  dataArrayJ->SetNumberOfTuples(arraySize);
  dataArrayJ->SetName("J");
  memcpy(dataArrayJ->GetVoidPointer(0), J, sizeof(double)*arraySize);

  delete ds;

  return 1;
}
