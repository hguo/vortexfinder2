#include "common/Inclusions.h"
#include <iostream>

int main(int argc, char **argv)
{
  if (argc < 2) return 1;

  const std::string filename(argv[1]);

  Inclusions inc;
  inc.ParseFromTextFile(filename);

  return 0;
}
