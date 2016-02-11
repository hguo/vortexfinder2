#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkFieldData.h"
#include "vtkRectilinearGrid.h"
#include "vtkStructuredGrid.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkBDATReader.h"
#include "io/GLGPU_IO_Helper.h"
#include "io/GLGPU3DDataset.h"

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

  GLHeader hdr;
  
  bool succ = false;
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadBDAT(
        FileName, hdr, NULL, NULL, NULL, NULL, true); 
  } 
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadLegacy(
        FileName, hdr, NULL, NULL, NULL, NULL, true); 
  }

  int ext[6] = {0, hdr.dims[0]-1, 0, hdr.dims[1]-1, 0, hdr.dims[2]-1};
  double cell_lengths[3] = {hdr.cell_lengths[0], hdr.cell_lengths[1], hdr.cell_lengths[2]},
         origins[3] = {hdr.origins[0], hdr.origins[1], hdr.origins[2]};

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), ext, 6);
  outInfo->Set(vtkDataObject::SPACING(), cell_lengths, 3);
  outInfo->Set(vtkDataObject::ORIGIN(), origins, 3);
  // vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, 1);

  return 1;
}

int vtkBDATReader::RequestData(
    vtkInformation*, 
    vtkInformationVector**, 
    vtkInformationVector* outVec)
{
  GLGPU3DDataset *ds = new GLGPU3DDataset;
  bool succ; 

  ds->OpenDataFileByPattern(FileName);
  // ds->SetPrecomputeSupercurrent(true);
  succ = ds->LoadTimeStep(0, 0);

  if (!succ) {
    vtkErrorMacro("Error opening file " << FileName);
    return 0;
  }

  GLHeader h;
  float *rho, *phi, *re, *im, *J;
  ds->GetDataArray(h, &rho, &phi, &re, &im, &J);

  // vtk data structures
  vtkInformation *outInfo = outVec->GetInformationObject(0);
  vtkImageData *imageData = 
    vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
  imageData->SetDimensions(h.dims[0], h.dims[1], h.dims[2]);
  // imageData->AllocateScalars(VTK_FLOAT, 1);

  // copy data
  const int arraySize = h.dims[0]*h.dims[1]*h.dims[2];
  vtkSmartPointer<vtkDataArray> dataArrayRho, dataArrayPhi, dataArrayRe, dataArrayIm, dataArrayJ;
  
  dataArrayRho.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayRho->SetNumberOfComponents(1); 
  dataArrayRho->SetNumberOfTuples(arraySize);
  dataArrayRho->SetName("rho");
  memcpy(dataArrayRho->GetVoidPointer(0), rho, sizeof(float)*arraySize);
  
  dataArrayPhi.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayPhi->SetNumberOfComponents(1); 
  dataArrayPhi->SetNumberOfTuples(arraySize);
  dataArrayPhi->SetName("phi");
  memcpy(dataArrayPhi->GetVoidPointer(0), phi, sizeof(float)*arraySize);

  dataArrayRe.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayRe->SetNumberOfComponents(1); 
  dataArrayRe->SetNumberOfTuples(arraySize);
  dataArrayRe->SetName("re");
  memcpy(dataArrayRe->GetVoidPointer(0), re, sizeof(float)*arraySize);
  
  dataArrayIm.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayIm->SetNumberOfComponents(1); 
  dataArrayIm->SetNumberOfTuples(arraySize);
  dataArrayIm->SetName("im");
  memcpy(dataArrayIm->GetVoidPointer(0), im, sizeof(float)*arraySize);

#if 0
  dataArrayJ.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayJ->SetNumberOfComponents(3);
  dataArrayJ->SetNumberOfTuples(arraySize);
  dataArrayJ->SetName("J");
  memcpy(dataArrayJ->GetVoidPointer(0), J, sizeof(float)*arraySize*3);
#endif

  imageData->GetPointData()->AddArray(dataArrayRho);
  imageData->GetPointData()->AddArray(dataArrayPhi);
  imageData->GetPointData()->AddArray(dataArrayRe);
  imageData->GetPointData()->AddArray(dataArrayIm);
  // imageData->GetPointData()->AddArray(dataArrayJ);

  // global attributes
  vtkSmartPointer<vtkDataArray> dataArrayB, dataArrayPBC, dataArrayJxext, dataArrayKx, dataArrayV;
  
  dataArrayB.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayB->SetNumberOfComponents(3);
  dataArrayB->SetNumberOfTuples(1);
  dataArrayB->SetName("B");
  dataArrayB->SetTuple(0, h.B);
 
  dataArrayPBC.TakeReference(vtkDataArray::CreateDataArray(VTK_UNSIGNED_CHAR));
  dataArrayPBC->SetNumberOfComponents(3);
  dataArrayPBC->SetNumberOfTuples(1);
  dataArrayPBC->SetName("pbc");
  dataArrayPBC->SetTuple3(0, h.pbc[0], h.pbc[1], h.pbc[2]);

  dataArrayJxext.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayJxext->SetNumberOfComponents(1);
  dataArrayJxext->SetNumberOfTuples(1);
  dataArrayJxext->SetName("Jxext");
  dataArrayJxext->SetTuple1(0, h.Jxext);
  
  dataArrayKx.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayKx->SetNumberOfComponents(1);
  dataArrayKx->SetNumberOfTuples(1);
  dataArrayKx->SetName("Kx");
  dataArrayKx->SetTuple1(0, h.Kex);
  
  dataArrayV.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayV->SetNumberOfComponents(1);
  dataArrayV->SetNumberOfTuples(1);
  dataArrayV->SetName("V");
  dataArrayV->SetTuple1(0, h.V);
  
  imageData->GetFieldData()->AddArray(dataArrayB);
  imageData->GetFieldData()->AddArray(dataArrayPBC);
  imageData->GetFieldData()->AddArray(dataArrayJxext);
  imageData->GetFieldData()->AddArray(dataArrayKx);
  imageData->GetFieldData()->AddArray(dataArrayV);

  delete ds;
  
  return 1;
}
