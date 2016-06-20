#include "vtkGLGPUSupercurrentFilter.h"
#include "vtkInformation.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "GL_post_process.h"
#include <cassert>

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
 
  int dims[3];
  double origins[3], cell_lengths[3], lengths[3];
  const int ndims = 3;
  inputData->GetDimensions(dims);
  inputData->GetOrigin(origins);
  inputData->GetSpacing(cell_lengths);

  double B[3], pbc1[3];
  bool pbc[3];
  dataArrayB->GetTuple(0, B);
  dataArrayPBC->GetTuple(0, pbc1);
  for (int i=0; i<3; i++) {
    pbc[i] = pbc1[i]; 
    lengths[i] = origins[i] + cell_lengths[i]*dims[i];
  }

  unsigned char btype = 0;
  if (pbc[0]) btype &= 0xff;
  if (pbc[1]) btype &= 0xff00;
  if (pbc[2]) btype &= 0xff0000;


  float Jxext = dataArrayJxext->GetTuple1(0);
  float Kex = dataArrayKx->GetTuple1(0);
  float V = dataArrayV->GetTuple1(0);

  fprintf(stderr, "B={%f, %f, %f}, pbc={%d, %d, %d}, Jxext=%f, Kx=%f, V=%f\n", 
      B[0], B[1], B[2], pbc[0], pbc[1], pbc[2], Jxext, Kex, V);

  const int arraySize = dims[0]*dims[1]*dims[2];
  float *rho = (float*)dataArrayRho->GetVoidPointer(0), 
        *phi = (float*)dataArrayPhi->GetVoidPointer(0), 
        *re = (float*)dataArrayRe->GetVoidPointer(0), 
        *im = (float*)dataArrayIm->GetVoidPointer(0);


  // GLPP
  GLPP *pp = new GLPP;
  // FIXME!
  pp->dim = ndims;
  pp->Nx = dims[0];
  pp->Ny = dims[1];
  pp->Nz = dims[2];
  pp->NN = arraySize;
  pp->btype = btype;
  pp->Lx = lengths[0];
  pp->Ly = lengths[1];
  pp->Lz = lengths[2];
  pp->dx = cell_lengths[0];
  pp->dy = cell_lengths[1];
  pp->dz = cell_lengths[2];
  pp->Bx = B[0];
  pp->By = B[1]; 
  pp->Bz = B[2];
  pp->KEx = Kex;
  pp->psi = (COMPLEX*)malloc(sizeof(COMPLEX)*arraySize);
  for (int i=0; i<arraySize; i++) {
    pp->psi[i].re = re[i];
    pp->psi[i].im = im[i];
  }

  pp->calc_current();
  assert(pp->Jx != NULL);

  float *J = (float*)malloc(sizeof(float)*arraySize*3);
  for (int i=0; i<arraySize; i++) {
    J[i*3] = pp->Jx[i];
    J[i*3+1] = pp->Jy[i];
    J[i*3+2] = pp->Jz[i];
  }
  
  delete pp;

  // output
  vtkSmartPointer<vtkDataArray> dataArrayJ; 
  dataArrayJ.TakeReference(vtkDataArray::CreateDataArray(VTK_FLOAT));
  dataArrayJ->SetNumberOfComponents(3); 
  dataArrayJ->SetNumberOfTuples(arraySize);
  dataArrayJ->SetName("J");
  memcpy(dataArrayJ->GetVoidPointer(0), J, sizeof(float)*arraySize*3);
  free(J);

  outputData->SetDimensions(dims[0], dims[1], dims[2]);
  outputData->GetPointData()->AddArray(dataArrayJ);

  return 1;
}
