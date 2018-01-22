#include <iostream>
#include "io/GLGPU_IO_Helper.h"

int main(int argc, char **argv)
{
  if (argc<3) return 1;
  const std::string filename_in = argv[1], 
                    filename_out = argv[2];
 
  int ndims = 3;
  int dims[3];
  float lengths[3];
  bool pbc[3];
  float time; 
  float B[3];
  float Jxext, Kx, V;
  float *re, *im;

  GLGPU_IO_Helper_ReadBDAT(
  // GLGPU_IO_Helper_ReadLegacy(
      filename_in, 
      ndims, dims, lengths, pbc,
      time, B, Jxext, Kx, V, 
      &re, &im);
  fprintf(stderr, "dims={%d, %d, %d}\n", dims[0], dims[1], dims[2]);

  GLGPU_IO_Helper_WriteNetCDF(
      filename_out, 
      ndims, dims, lengths, pbc,
      B, Jxext, Kx, V, 
      re, im);

  return 0;
}
