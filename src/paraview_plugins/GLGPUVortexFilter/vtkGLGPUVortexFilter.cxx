#include "vtkGLGPUVortexFilter.h"
#include "vtkInformation.h"
#include "vtkSmartPointer.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyLine.h"
#include "vtkCellArray.h"
#include "vtkImageData.h"
#include "vtkSphereSource.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "io/GLGPU3DDataset.h"
#include "extractor/Extractor.h"

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

  return ExtractVorticies(input, output);
}

int vtkGLGPUVortexFilter::ExtractVorticies(vtkImageData* imageData, vtkPolyData* polyData)
{
  // TODO: check compatability
  vtkSmartPointer<vtkDataArray> dataArrayRho, dataArrayPhi, dataArrayRe, dataArrayIm;
  vtkSmartPointer<vtkDataArray> dataArrayB, dataArrayPBC, dataArrayJxext, dataArrayKx, dataArrayV;
  int index;

  dataArrayRho = imageData->GetPointData()->GetArray("rho", index);
  dataArrayPhi = imageData->GetPointData()->GetArray("phi", index);
  dataArrayRe = imageData->GetPointData()->GetArray("re", index);
  dataArrayIm = imageData->GetPointData()->GetArray("im", index);
  dataArrayB = imageData->GetFieldData()->GetArray("B", index);
  dataArrayPBC = imageData->GetFieldData()->GetArray("pbc", index);
  dataArrayJxext = imageData->GetFieldData()->GetArray("Jxext", index);
  dataArrayKx = imageData->GetFieldData()->GetArray("Kx", index);
  dataArrayV = imageData->GetFieldData()->GetArray("V", index);

  // fprintf(stderr, "dataType=%d\n", dataArrayRho->GetDataType());

  GLHeader h;
  h.ndims = 3;
  imageData->GetDimensions(h.dims);
  imageData->GetOrigin(h.origins);
  imageData->GetSpacing(h.cell_lengths);
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
  //     h.B[0], h.B[1], h.B[2], h.pbc[0], h.pbc[1], h.pbc[2], h.Jxext, h.Kex, h.V);

  const int count = h.dims[0]*h.dims[1]*h.dims[2];
  double *rho = (double*)dataArrayRho->GetVoidPointer(0), 
         *phi = (double*)dataArrayPhi->GetVoidPointer(0), 
         *re = (double*)dataArrayRe->GetVoidPointer(0), 
         *im = (double*)dataArrayIm->GetVoidPointer(0);

  // build data
  GLGPU3DDataset *ds = new GLGPU3DDataset;
  ds->BuildDataFromArray(h, rho, phi, re, im); // FIXME
  // ds->SetMeshType(GLGPU3D_MESH_TET);
  ds->SetMeshType(GLGPU3D_MESH_HEX);
  ds->BuildMeshGraph();

  VortexExtractor *ex = new VortexExtractor;
  ex->SetDataset(ds);
  ex->SetArchive(false);
#if WITH_CUDA
  ex->SetGPU(true); // FIXME: failure fallback
#endif
  ex->SetGaugeTransformation(true); 
  ex->ExtractFaces(0);
  ex->TraceOverSpace(0);

  std::vector<VortexLine> vlines = ex->GetVortexLines(0);
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
  vtkSmartPointer<vtkCellArray> cells = vtkCellArray::New();

  std::vector<int> vertCounts;
  for (int i=0; i<vlines.size(); i++) {
    int vertCount = 0;
    const int nv = vlines[i].size()/3;
    // if (nv<2) continue;
    double p0[3];
    for (int j=0; j<nv; j++) {
      double p[3] = {vlines[i][j*3], vlines[i][j*3+1], vlines[i][j*3+2]};
      points->InsertNextPoint(p);

      double delta[3] = {p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
      double dist = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);

      if (j>0 && dist>5) { // FIXME
        vertCounts.push_back(vertCount);
        vertCount = 0;
      }
      memcpy(p0, p, sizeof(double)*3);
      vertCount ++;
    }

    if (vertCount > 0) 
      vertCounts.push_back(vertCount);
  }

  int nv = 0;
  for (int i=0; i<vertCounts.size(); i++) {
    // fprintf(stderr, "vertCount=%d\n", vertCounts[i]);
    vtkSmartPointer<vtkPolyLine> polyLine = vtkPolyLine::New();
    polyLine->GetPointIds()->SetNumberOfIds(vertCounts[i]);
    for (int j=0; j<vertCounts[i]; j++)
      polyLine->GetPointIds()->SetId(j, j+nv);

    cells->InsertNextCell(polyLine);
    nv += vertCounts[i];
  }

  polyData->SetPoints(points);
  polyData->SetLines(cells);

  delete ex;
  delete ds;

  return 1;
}
