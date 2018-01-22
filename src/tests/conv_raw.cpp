#include <iostream>
#include <cassert>
#include "io/GLGPU_IO_Helper.h"

int main(int argc, char **argv)
{
  if (argc<4) return 1;
  const std::string filename_in = argv[1], 
                    filename_out_re = argv[2],
                    filename_out_im = argv[3];
 
  GLHeader h;
  float *rho, *phi, *re, *im;

  // assert( GLGPU_IO_Helper_ReadBDAT(filename_in, h, &rho, &phi, &re, &im, NULL, NULL, NULL, false, false) );
  GLGPU_IO_Helper_ReadLegacy(filename_in, h, &rho, &phi, &re, &im, NULL, NULL, NULL, false, false);

  size_t num_elems = h.dims[0] * h.dims[1] * h.dims[2];
  fprintf(stderr, "%d, %d, %d\n", h.dims[0], h.dims[1], h.dims[2]);
  
  FILE *fp_re = fopen(filename_out_re.c_str(), "wb");
  fwrite(re, sizeof(float), num_elems, fp_re);
  fclose(fp_re);

  FILE *fp_im = fopen(filename_out_im.c_str(), "wb");
  fwrite(im, sizeof(float), num_elems, fp_im);
  fclose(fp_im);

  free(rho);
  free(phi);
  free(re);
  free(im);

  return 0;
}
