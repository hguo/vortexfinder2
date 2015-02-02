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
  void ExtractFace(const Face*);
  void ExtractFacePrism(const Face*);

  int CheckFace(double X[3][3], double A[3][3], double re[3], double im[3]) const; // returns chirality
  int CheckVirtualFace(double X[2][3], double A[4][3], double re[4], double im[4]) const; // return chirality

protected:
  PuncturedElem* NewPuncturedElem(ElemIdType) const;
  
  bool FindZero(const double X[][3], const double re[], const double im[], double pos[3]) const;
}; 

#endif
