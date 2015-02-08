#ifndef _PUNCTURE_H
#define _PUNCTURE_H

struct CPuncturedFace
{
  CFace *face;
  int chirality;
  std::vector<double> pos;
};

struct CPuncturedEdge
{
  CEdge *edge;
  int chirality; 
  double t; // punctured time
};

struct PuncturedElem
{
  CElem *elem;
  std::vector<int> chiralities; // chiralities on faces
};

#endif
