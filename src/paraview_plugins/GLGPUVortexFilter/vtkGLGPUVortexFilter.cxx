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

#if 0
  // dummy output
  vtkSmartPointer<vtkSphereSource> source = vtkSphereSource::New();
  source->SetRadius(30.0);
  source->SetCenter(0, 0, 0);
  source->Update();

  output->DeepCopy(source->GetOutput());
  
  return 1;
#endif
}

int vtkGLGPUVortexFilter::ExtractVorticies(vtkImageData* imageData, vtkPolyData* polyData)
{
  // TODO: check compatability
  vtkSmartPointer<vtkDataArray> dataArrayRe, dataArrayIm;
  int index;

  dataArrayRe = imageData->GetPointData()->GetArray("re", index);
  dataArrayIm = imageData->GetPointData()->GetArray("im", index);

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

  bool pbc[3] = {1, 0, 0}; 
  double time = 0; // dummy
  double B[3] = {0.13, 0, 0}; // TODO
  double Jxext;
  double Kx = 0; // TODO
  double V;

  GLGPU3DDataset *ds = new GLGPU3DDataset;
  ds->BuildDataFromArray(
      ndims, dims, lengths, pbc, time, B, Jxext, Kx, V, 
      (double*)dataArrayRe->GetVoidPointer(0),
      (double*)dataArrayIm->GetVoidPointer(0));
  ds->BuildMeshGraph();

  GLGPUVortexExtractor *ex = new GLGPUVortexExtractor;
  ex->SetDataset(ds);
  ex->SetArchive(false);
  ex->SetGaugeTransformation(false); 
  ex->ExtractFaces(0);
  ex->TraceOverSpace(0);

  std::vector<VortexLine> vlines = ex->GetVortexLines(0);
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
  vtkSmartPointer<vtkCellArray> cells = vtkCellArray::New();
  int np = 0;

  for (int i=0; i<vlines.size(); i++) {
    const int nv = vlines[i].size()/3;
    if (nv<=0) continue;
    // fprintf(stderr, "id=%d, n=%d\n", i, nv);
    for (int j=0; j<nv; j++) {
      double p[3] = {vlines[i][j*3], vlines[i][j*3+1], vlines[i][j*3+2]};
      // fprintf(stderr, "%f, %f, %f\n", p[0], p[1], p[2]);
      points->InsertNextPoint(p);
    }

    vtkSmartPointer<vtkPolyLine> polyLine = vtkPolyLine::New();
    polyLine->GetPointIds()->SetNumberOfIds(nv);
    for (int j=0; j<nv; j++)
      polyLine->GetPointIds()->SetId(j, j+np);

    cells->InsertNextCell(polyLine);
    np+=nv;
  }

  polyData->SetPoints(points);
  polyData->SetLines(cells);

  // TODO: transform output data to polyData;

  delete ex;
  delete ds;

  return 1;
}
