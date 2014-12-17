#include <iostream>
#include "io/BDATReader.h"

int main(int argc, char **argv)
{
  if (argc<2) return 1;
  BDATReader reader(argv[1]);

  fprintf(stderr, "%d\n", reader.Valid());

  return 0;
}
