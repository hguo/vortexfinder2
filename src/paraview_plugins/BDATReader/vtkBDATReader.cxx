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

  int ext[6] = {0, 10, 0, 10, 0, 10};
  double spacing[3] = {1, 1, 1};
  double origin[3] = {0, 0, 0};

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), ext, 6);
  outInfo->Set(vtkDataObject::SPACING(), spacing, 3);
  outInfo->Set(vtkDataObject::ORIGIN(), origin, 3);
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);

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
  
  int extent[6] = {0, dims[0], 0, dims[1], 0, dims[2]};
  // outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent);
  imageData->SetExtent(extent);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  imageData->SetDimensions(dims[0], dims[1], dims[2]);
  imageData->AllocateScalars(VTK_DOUBLE, 1);
  vtkDataArray *scalars = imageData->GetPointData()->GetScalars();
  scalars->SetName("re");

  for (int i=0; i<dims[0]; i++) 
    for (int j=0; j<dims[1]; j++)
      for (int k=0; k<dims[2]; k++) {
        double *pixel = static_cast<double*>(imageData->GetScalarPointer(i, j, k));
        pixel[0] = 2.0*cos(i);
        // int idx = i + dims[0] * (j + dims[1] * k);
      }

  free(re);
  free(im);
  
  return 1;
}
