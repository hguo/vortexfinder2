#ifndef _CONDOR2EXTRACTOR_H
#define _CONDOR2EXTRACTOR_H

#include <string>
#include <map>
#include "Extractor.h"
#include "PuncturedElem.h"
#include "io/Condor2Dataset.h"

/* 
 * \class   Condor2VortexExtractor
 * \author  Hanqi Guo
 * \brief   Vortex extractor for Condor2 output
*/
class Condor2VortexExtractor : public VortexExtractor
{
public:
  Condor2VortexExtractor(); 
  ~Condor2VortexExtractor();

  void SetDataset(const GLDataset* ds);
  const Condor2Dataset* Dataset() const {return _ds;}

  void Extract();

private: 
  int _verbose; 
  bool _gauge; 
 
  const Condor2Dataset *_ds;
}; 

#endif
