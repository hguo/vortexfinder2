#ifndef __vtkBDATSeriesReader_h
#define __vtkBDATSeriesReader_h

#include "vtkDataReader.h"
#include "vtkDataObjectAlgorithm.h"
#include "vtkImageAlgorithm.h"

class vtkBDATSeriesReader : public vtkImageAlgorithm
{
public:
  static vtkBDATSeriesReader *New();
  vtkTypeMacro(vtkBDATSeriesReader, vtkImageAlgorithm);

  // vtkSetStringMacro(FileName);
  // vtkGetStringMacro(FileName);

  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

protected:
  vtkBDATSeriesReader();
  ~vtkBDATSeriesReader();

private:
  vtkBDATSeriesReader(const vtkBDATSeriesReader&);
  void operator=(const vtkBDATSeriesReader&);
};

#endif
