#ifndef _CONDOR2EXTRACTOR_H
#define _CONDOR2EXTRACTOR_H

#include <string>
#include <map>
#include "Extractor.h"
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
  void ExtractFace(FaceIdType, int time); // time=0 or 1
  void ExtractSpaceTimeEdge(EdgeIdType);

  ChiralityType CheckFace(const double X[3][3], const double A[3][3], const double re[3], const double im[3], double pos[3]) const; // returns chirality
  ChiralityType CheckSpaceTimeEdge(const double X[2][3], const double A[4][3], const double re[4], const double im[4], double &t) const; // return chirality

protected:
  bool FindFaceZero(const double X[][3], const double re[], const double im[], double pos[3]) const;
}; 

#endif
