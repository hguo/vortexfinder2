#ifndef vtkBDATSeriesReader_h
#define vtkBDATSeriesReader_h

#include "vtkVortexFiltersModule.h"
#include "vtkDataReader.h"
#include "vtkDataObjectAlgorithm.h"
#include "vtkImageAlgorithm.h"
#include <string>
#include <vector>
#include <map>

class VTKVORTEXFILTERS_EXPORT vtkBDATSeriesReader : public vtkImageAlgorithm
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
  std::vector<double> TimeSteps;
  std::map<double, int> TimeStepsMap;
  vtkIdType FileIndex;

protected:
  vtkBDATSeriesReader();
  ~vtkBDATSeriesReader();

private:
  vtkBDATSeriesReader(const vtkBDATSeriesReader&);
  void operator=(const vtkBDATSeriesReader&);
};

#endif
