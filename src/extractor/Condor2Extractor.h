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

  void Extract();

protected:
  PuncturedElem* NewPuncturedElem(ElemIdType) const;
  
  bool FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const;
}; 

#endif
