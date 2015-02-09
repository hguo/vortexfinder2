#ifndef _PUNCTURE_H
#define _PUNCTURE_H

struct PuncturedFace
{
  int chirality;
  double pos[3];
  
  bool visited; 

  PuncturedFace() : visited(false) {}
};

struct PuncturedEdge
{
  int chirality; 
  double t; // punctured time
  
  bool visited; 

  PuncturedEdge() : visited(false) {}
};

struct PuncturedElem
{
  int chiralities[6]; // chiralities on faces
};

#endif
