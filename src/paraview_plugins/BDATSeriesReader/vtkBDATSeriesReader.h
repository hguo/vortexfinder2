#ifndef __vtkBDATSeriesReader_h
#define __vtkBDATSeriesReader_h

#include "vtkDataReader.h"
#include "vtkDataObjectAlgorithm.h"
#include "vtkImageAlgorithm.h"
#include <string>
#include <vector>

class vtkBDATSeriesReader : public vtkImageAlgorithm
{
public:
  static vtkBDATSeriesReader *New();
  vtkTypeMacro(vtkBDATSeriesReader, vtkImageAlgorithm);

  virtual void AddFileName(const char*);
  virtual void RemoveAllFileNames();
  virtual unsigned int GetNumberOfFileNames();
  virtual const char* GetFileName(unsigned int idx);
  const char* GetCurrentFileName();

  vtkSetMacro(FileIndex, vtkIdType);
  vtkGetMacro(FileIndex, vtkIdType);

  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

private:
  std::vector<std::string> FileNames;
  vtkIdType FileIndex; 

protected:
  vtkBDATSeriesReader();
  ~vtkBDATSeriesReader();

private:
  vtkBDATSeriesReader(const vtkBDATSeriesReader&);
  void operator=(const vtkBDATSeriesReader&);
};

#endif
