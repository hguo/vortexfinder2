#ifndef __vtkBDATReader_h
#define __vtkBDATReader_h

#include "vtkDataReader.h"
#include "vtkDataObjectAlgorithm.h"
#include "vtkImageAlgorithm.h"

class vtkBDATReader : public vtkImageAlgorithm
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

private:
  vtkBDATReader(const vtkBDATReader&);
  void operator=(const vtkBDATReader&);
};

#endif
