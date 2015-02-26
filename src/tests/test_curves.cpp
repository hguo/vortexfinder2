#include "fitCurves/fitCurves.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  using namespace FitCurves;
 
  const int num = 10;
  Point<3> *pts = (Point<3>*)malloc(num*sizeof(Point<3>));
  for (int i=0; i<num; i++) {
    pts[i][0] = (float)rand()/RAND_MAX;
    pts[i][1] = (float)rand()/RAND_MAX;
    pts[i][2] = (float)rand()/RAND_MAX;
  }

  Point<3> *curve = (Point<3>*)malloc(num*4*sizeof(Point<3>));

  const double error_bound = 0.1;
  double sum_error;
  size_t np = fit_curves(num, pts, error_bound, curve, sum_error);

  // fprintf(stderr, "np=%d\n", np);

  free(curve);
  free(pts);

  return 0;
}
