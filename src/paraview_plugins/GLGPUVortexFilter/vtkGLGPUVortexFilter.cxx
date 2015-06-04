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
#include "extractor/GLGPUExtractor.h"

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
  vtkSmartPointer<vtkDataArray> dataArrayRe, dataArrayIm;
  vtkSmartPointer<vtkDataArray> dataArrayB, dataArrayPBC, dataArrayJxext, dataArrayKx, dataArrayV;
  int index;

  dataArrayRe = imageData->GetPointData()->GetArray("re", index);
  dataArrayIm = imageData->GetPointData()->GetArray("im", index);
  dataArrayB = imageData->GetFieldData()->GetArray("B", index);
  dataArrayPBC = imageData->GetFieldData()->GetArray("pbc", index);
  dataArrayJxext = imageData->GetFieldData()->GetArray("Jxext", index);
  dataArrayKx = imageData->GetFieldData()->GetArray("Kx", index);
  dataArrayV = imageData->GetFieldData()->GetArray("V", index);

  const int ndims = 3;
  int dims[3];
  imageData->GetDimensions(dims);

  double origins[3];
  imageData->GetOrigin(origins);

  double cellLengths[3];
  imageData->GetSpacing(cellLengths);

  double lengths[3];
  for (int i=0; i<3; i++) 
    lengths[i] = cellLengths[i] * dims[i];

  bool pbc[3];
  double pbc1[3];
  double time = 0; // FIXME
  double B[3];
  double Jxext;
  double Kx;
  double V;

  dataArrayB->GetTuple(0, B);
  dataArrayPBC->GetTuple(0, pbc1);
  for (int i=0; i<3; i++)
    pbc[i] = (pbc1[i]>0);
  Jxext = dataArrayJxext->GetTuple1(0);
  Kx = dataArrayKx->GetTuple1(0);
  V = dataArrayV->GetTuple1(0);

  // fprintf(stderr, "B={%f, %f, %f}, pbc={%d, %d, %d}, Jxext=%f, Kx=%f, V=%f\n", 
  //     B[0], B[1], B[2], pbc[0], pbc[1], pbc[2], Jxext, Kx, V);

  // build data
  GLGPU3DDataset *ds = new GLGPU3DDataset;
  ds->BuildDataFromArray(
      ndims, dims, lengths, pbc, time, B, Jxext, Kx, V, 
      (double*)dataArrayRe->GetVoidPointer(0),
      (double*)dataArrayIm->GetVoidPointer(0));
  ds->BuildMeshGraph();

  GLGPUVortexExtractor *ex = new GLGPUVortexExtractor;
  ex->SetDataset(ds);
  ex->SetArchive(false);
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
    if (nv==0) continue;
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
