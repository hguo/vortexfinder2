#include <iostream>
#include "io/GLGPU_IO_Helper.h"

int main(int argc, char **argv)
{
  if (argc<3) return 1;
  const std::string filename_in = argv[1], 
                    filename_out = argv[2];
 
  float *rho, *phi, *re, *im, *Jx, *Jy, *Jz;
  GLHeader h;

  GLGPU_IO_Helper_ReadLegacy(filename_in, h, &rho, &phi, &re, &im, &Jx, &Jy, &Jz, false, true); 
  // GLGPU_IO_Helper_ReadBDAT(filename_in, ndims, dims, lengths, pbc, time, B, Jxext, Kx, V, &re, &im);

  GLGPU_IO_Helper_WriteNetCDF(filename_out, h, rho, phi, re, im, Jx, Jy, Jz);

  return 0;
}
