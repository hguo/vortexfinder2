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

int vtkGLGPUSupercurrentFilter::RequestUpdateExtent(
    vtkInformation*, 
    vtkInformationVector** inputVector, 
    vtkInformationVector*)
{
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), 
      inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()), 6);

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

  output->SetExtent(
      outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()));

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
  double origins[3], cell_lengths[3];

  h.ndims = 3;
  inputData->GetDimensions(h.dims);
  inputData->GetOrigin(origins);
  inputData->GetSpacing(cell_lengths);
  for (int i=0; i<3; i++) {
    h.origins[i] = origins[i];
    h.cell_lengths[i] = cell_lengths[i];
    h.lengths[i] = h.cell_lengths[i] * h.dims[i];
  }

  double B[3], pbc1[3];
  dataArrayB->GetTuple(0, B);
  dataArrayPBC->GetTuple(0, pbc1);
  for (int i=0; i<3; i++) {
    // h.pbc[i] = (pbc1[i]>0);
    h.pbc[i] = 0; 
    h.B[i] = B[i];
  }

  h.Jxext = dataArrayJxext->GetTuple1(0);
  h.Kex = dataArrayKx->GetTuple1(0);
  h.V = dataArrayV->GetTuple1(0);

  // fprintf(stderr, "B={%f, %f, %f}, pbc={%d, %d, %d}, Jxext=%f, Kx=%f, V=%f\n", 
  //     h.B[0], h.B[1], h.B[2], h.pbc[0], h.pbc[1], h.pbc[2], h.Jxext, h.Kex, h.V);

  const int arraySize = h.dims[0]*h.dims[1]*h.dims[2];
  float *rho = (float*)dataArrayRho->GetVoidPointer(0), 
        *phi = (float*)dataArrayPhi->GetVoidPointer(0), 
        *re = (float*)dataArrayRe->GetVoidPointer(0), 
        *im = (float*)dataArrayIm->GetVoidPointer(0);

  // build data
  GLGPU3DDataset *ds = new GLGPU3DDataset;
  ds->BuildDataFromArray(h, rho, phi, re, im);
  ds->ComputeSupercurrentField();
  const float *J = ds->GetSupercurrentDataArray();

  vtkSmartPointer<vtkDataArray> dataArrayJ; 
  dataArrayJ.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayJ->SetNumberOfComponents(3); 
  dataArrayJ->SetNumberOfTuples(arraySize);
  dataArrayJ->SetName("J");
  memcpy(dataArrayJ->GetVoidPointer(0), J, sizeof(float)*arraySize*3);

  outputData->SetDimensions(h.dims[0], h.dims[1], h.dims[2]);
  outputData->GetPointData()->AddArray(dataArrayJ);

  delete ds;

  return 1;
}
