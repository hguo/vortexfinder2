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
#include "vtkBDATSeriesReader.h"
#include "io/GLGPU_IO_Helper.h"
#include <assert.h>

vtkStandardNewMacro(vtkBDATSeriesReader);

vtkBDATSeriesReader::vtkBDATSeriesReader()
{
  SetNumberOfInputPorts(0);
  SetNumberOfOutputPorts(1);
}

vtkBDATSeriesReader::~vtkBDATSeriesReader()
{
}

void vtkBDATSeriesReader::AddFileName(const char* filename)
{
  FileNames.push_back(filename);
}

void vtkBDATSeriesReader::RemoveAllFileNames()
{
  FileNames.clear();
}

const char* vtkBDATSeriesReader::GetFileName(unsigned int idx)
{
  if (idx >= FileNames.size())
    return 0;
  else 
    return FileNames[idx].c_str();
}

const char* vtkBDATSeriesReader::GetCurrentFileName()
{
  return GetFileName(FileIndex);
}

unsigned int vtkBDATSeriesReader::GetNumberOfFileNames()
{
  return FileNames.size();
}

int vtkBDATSeriesReader::RequestInformation(
    vtkInformation* request,
    vtkInformationVector**, 
    vtkInformationVector* outVec)
{
  vtkInformation *outInfo = outVec->GetInformationObject(0);

  const int nfiles = GetNumberOfFileNames();
  TimeSteps.clear();
  TimeStepsMap.clear();

  GLHeader h;

  for (int fidx=0; fidx<nfiles; fidx++) {
    bool succ = false;
    if (!succ) {
      succ = GLGPU_IO_Helper_ReadBDAT(
          FileNames[fidx], h, NULL, true);
    } 
    if (!succ) {
      succ = GLGPU_IO_Helper_ReadLegacy(
          FileNames[fidx], h, NULL, true);
    }
    if (!succ) {
      fprintf(stderr, "cannot open file: %s\n", FileNames[fidx].c_str());
      assert(false); // TODO
    }

    TimeSteps.push_back(h.time);
    TimeStepsMap[h.time] = fidx;
    // fprintf(stderr, "fidx=%d, time=%f\n", fidx, time);
 
    if (fidx == 0) {
      int ext[6] = {0, h.dims[0]-1, 0, h.dims[1]-1, 0, h.dims[2]-1};

      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), ext, 6);
      outInfo->Set(vtkDataObject::SPACING(), h.cell_lengths, 3);
      outInfo->Set(vtkDataObject::ORIGIN(), h.origins, 3);
      // vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_DOUBLE, 1);
    }
  }

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(), 
      &TimeSteps[0], static_cast<int>(TimeSteps.size()));

  return 1;
}

int vtkBDATSeriesReader::RequestData(
    vtkInformation*, 
    vtkInformationVector**, 
    vtkInformationVector* outVec)
{
  vtkInformation *outInfo = outVec->GetInformationObject(0);
  double upTime = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
  int upTimeStep = TimeStepsMap[upTime];
  std::string filename = FileNames[upTimeStep];

  fprintf(stderr, "uptime=%f, timestep=%d\n", upTime, upTimeStep);

  GLHeader h;
  double *rho=NULL, *phi=NULL;
  double *psi;

  bool succ = false;
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadBDAT(
        filename.c_str(), h, &psi);
  }
  if (!succ) {
    succ = GLGPU_IO_Helper_ReadLegacy(
        filename.c_str(), h, &psi);
  }
  if (!succ)
  {
    vtkErrorMacro("Error opening file " << filename);
    return 0;
  }
    
  // vtk data structures
  vtkImageData *imageData = 
    vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
  imageData->SetDimensions(h.dims[0], h.dims[1], h.dims[2]);
  // imageData->AllocateScalars(VTK_DOUBLE, 1);

  // copy data
  const int arraySize = h.dims[0]*h.dims[1]*h.dims[2];
  vtkSmartPointer<vtkDataArray> dataArrayRe, dataArrayIm, dataArrayRho, dataArrayPhi;

  rho = (double*)malloc(sizeof(double)*arraySize);
  phi = (double*)malloc(sizeof(double)*arraySize);
  for (int i=0; i<arraySize; i++) {
    rho[i] = psi[i*2];
    phi[i] = psi[i*2+1];
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

#if 0
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
#endif

  imageData->GetPointData()->AddArray(dataArrayRho);
  imageData->GetPointData()->AddArray(dataArrayPhi);
  imageData->GetPointData()->AddArray(dataArrayRe);
  imageData->GetPointData()->AddArray(dataArrayIm);

  // global attributes
  vtkSmartPointer<vtkDataArray> dataArrayB, dataArrayPBC, dataArrayJxext, dataArrayKx, dataArrayV;
  
  dataArrayB.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayB->SetNumberOfComponents(3);
  dataArrayB->SetNumberOfTuples(1);
  dataArrayB->SetName("B");
  dataArrayB->SetTuple(0, h.B);
 
  dataArrayPBC.TakeReference(vtkDataArray::CreateDataArray(VTK_UNSIGNED_CHAR));
  dataArrayPBC->SetNumberOfComponents(3);
  dataArrayPBC->SetNumberOfTuples(1);
  dataArrayPBC->SetName("pbc");
  dataArrayPBC->SetTuple3(0, h.pbc[0], h.pbc[1], h.pbc[2]);

  dataArrayJxext.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayJxext->SetNumberOfComponents(1);
  dataArrayJxext->SetNumberOfTuples(1);
  dataArrayJxext->SetName("Jxext");
  dataArrayJxext->SetTuple1(0, h.Jxext);
  
  dataArrayKx.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayKx->SetNumberOfComponents(1);
  dataArrayKx->SetNumberOfTuples(1);
  dataArrayKx->SetName("Kx");
  dataArrayKx->SetTuple1(0, h.Kex);
  
  dataArrayV.TakeReference(vtkDataArray::CreateDataArray(VTK_DOUBLE));
  dataArrayV->SetNumberOfComponents(1);
  dataArrayV->SetNumberOfTuples(1);
  dataArrayV->SetName("V");
  dataArrayV->SetTuple1(0, h.V);
  
  imageData->GetFieldData()->AddArray(dataArrayB);
  imageData->GetFieldData()->AddArray(dataArrayPBC);
  imageData->GetFieldData()->AddArray(dataArrayJxext);
  imageData->GetFieldData()->AddArray(dataArrayKx);
  imageData->GetFieldData()->AddArray(dataArrayV);

  free(rho);
  free(phi);

  return 1;
}
