#ifndef vtkBDATReader_h
#define vtkBDATReader_h

#include "vtkVortexFiltersModule.h"
#include "vtkDataReader.h"
#include "vtkDataObjectAlgorithm.h"
#include "vtkImageAlgorithm.h"

class VTKVORTEXFILTERS_EXPORT vtkBDATReader : public vtkImageAlgorithm
{
public:
  static vtkBDATReader *New();
  vtkTypeMacro(vtkBDATReader, vtkImageAlgorithm);

  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*);
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*);

protected:
  vtkBDATReader();
  ~vtkBDATReader();

  char *FileName;
  int GLdim;

private:
  vtkBDATReader(const vtkBDATReader&);
  void operator=(const vtkBDATReader&);
};

#endif
