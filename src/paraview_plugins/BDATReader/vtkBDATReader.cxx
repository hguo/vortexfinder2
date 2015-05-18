#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkRectilinearGrid.h"
#include "vtkStructuredGrid.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkBDATReader.h"
#include "io/GLGPU_IO_Helper.h"

vtkStandardNewMacro(vtkBDATReader);

vtkBDATReader::vtkBDATReader()
{
  FileName = NULL;
  SetNumberOfInputPorts(0);
  SetNumberOfOutputPorts(1);
}

vtkBDATReader::~vtkBDATReader()
{
  SetFileName(0);
}

int vtkBDATReader::RequestInformation(
    vtkInformation*, 
    vtkInformationVector**, 
    vtkInformationVector* outVec)
{
  vtkInformation *outInfo = outVec->GetInformationObject(0);

  int ndims; 
  int dims[3];
  bool pbc[3];
  double origins[3], lengths[3], cellLengths[3], B[3];
  double time, Jxext, Kx, V;
  
  bool succ = false;
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadBDAT(
        FileName, ndims, dims, lengths, pbc, 
        time, B, Jxext, Kx, V, NULL, NULL, true); // header only
  } 
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadLegacy(
        FileName, ndims, dims, lengths, pbc, 
        time, B, Jxext, Kx, V, NULL, NULL, true);
  }
 
  for (int i=0; i<ndims; i++) {
    origins[i] = -0.5*lengths[i];
    cellLengths[i] = lengths[i] / (dims[i]-1);
  }

  int ext[6] = {0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1};

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), ext, 6);
  outInfo->Set(vtkDataObject::SPACING(), cellLengths, 3);
  outInfo->Set(vtkDataObject::ORIGIN(), origins, 3);
  // vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);

  return 1;
}

int vtkBDATReader::RequestData(
    vtkInformation*, 
    vtkInformationVector**, 
    vtkInformationVector* outVec)
{
  // load the data
  int ndims; 
  int dims[3];
  bool pbc[3];
  double origins[3], lengths[3], cellLengths[3], B[3];
  double time, Jxext, Kx, V;
  double *re=NULL, *im=NULL;
  double *rho=NULL, *phi=NULL;

  bool succ = false;
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadBDAT(
        FileName, ndims, dims, lengths, pbc, 
        time, B, Jxext, Kx, V, &re, &im);
  }
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadLegacy(
        FileName, ndims, dims, lengths, pbc, 
        time, B, Jxext, Kx, V, &re, &im);
  }
  if (!succ || ndims!=3)
  {
    vtkErrorMacro("Error opening file " << FileName);
    return 0;
  }

  for (int i=0; i<ndims; i++) {
    origins[i] = -0.5*lengths[i];
    cellLengths[i] = lengths[i] / (dims[i]-1);
  }
    
  // vtk data structures
  vtkInformation *outInfo = outVec->GetInformationObject(0);
  vtkImageData *imageData = 
    vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
  imageData->SetDimensions(dims[0], dims[1], dims[2]);
  // imageData->AllocateScalars(VTK_DOUBLE, 1);

  const int arraySize = dims[0]*dims[1]*dims[2];
  vtkSmartPointer<vtkDataArray> dataArrayRe, dataArrayIm, dataArrayRho, dataArrayPhi;

  rho = (double*)malloc(sizeof(double)*arraySize);
  phi = (double*)malloc(sizeof(double)*arraySize);
  for (int i=0; i<arraySize; i++) {
    rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);
    phi[i] = atan2(im[i], re[i]);
  }
  
  dataArrayRho.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayRho->SetNumberOfComponents(1); 
  dataArrayRho->SetNumberOfTuples(arraySize);
  dataArrayRho->SetName("rho");
  memcpy(dataArrayRho->GetVoidPointer(0), rho, sizeof(double)*arraySize);
  
  dataArrayPhi.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayPhi->SetNumberOfComponents(1); 
  dataArrayPhi->SetNumberOfTuples(arraySize);
  dataArrayPhi->SetName("phi");
  memcpy(dataArrayPhi->GetVoidPointer(0), phi, sizeof(double)*arraySize);

  dataArrayRe.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayRe->SetNumberOfComponents(1); 
  dataArrayRe->SetNumberOfTuples(arraySize);
  dataArrayRe->SetName("re");
  memcpy(dataArrayRe->GetVoidPointer(0), re, sizeof(double)*arraySize);
  
  dataArrayIm.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayIm->SetNumberOfComponents(1); 
  dataArrayIm->SetNumberOfTuples(arraySize);
  dataArrayIm->SetName("im");
  memcpy(dataArrayIm->GetVoidPointer(0), im, sizeof(double)*arraySize);

  imageData->GetPointData()->AddArray(dataArrayRho);
  imageData->GetPointData()->AddArray(dataArrayPhi);
  imageData->GetPointData()->AddArray(dataArrayRe);
  imageData->GetPointData()->AddArray(dataArrayIm);

  // TODO: global properties, including B, V, etc.

  free(rho);
  free(phi);
  free(re);
  free(im);
  
  return 1;
}
