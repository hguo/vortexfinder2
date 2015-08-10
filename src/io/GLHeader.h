#ifndef _GLHEADER_H
#define _GLHEADER_H

typedef struct {
  int ndims; 
  int dims[3];
  bool pbc[3];
  double lengths[3], origins[3], cell_lengths[3];
  double time;
  double B[3];
  double Jxext, Kex, Kex_dot, V;
  double fluctuation_amp;
} GLHeader;

#endif
