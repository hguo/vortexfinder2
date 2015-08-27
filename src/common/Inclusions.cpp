#include "Inclusions.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

Inclusions::Inclusions()
{
}

Inclusions::~Inclusions()
{
}

void Inclusions::PrintInfo()
{
}

void Inclusions::Clear()
{
  _x.clear();
  _y.clear();
  _z.clear();
}

bool Inclusions::ParseFromTextFile(const std::string& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open()) return false;

  int n = 0;
  int num_inclusions;
  double x, y, z;

  Clear();

  std::string line;
  while (getline(ifs, line)) {
    if (line.length() == 0 || line[0] == '#' || line[0] == ' ')
      continue;
    if (n == 0 || n == 1) { // skip line 1 and line 2 
      n++; 
      continue;
    }
   
    std::stringstream ss(line);
    if (n == 2) {
      ss >> num_inclusions; 
      n++; 
      continue;
    }

    ss >> x >> y >> z;

    _x.push_back(x);
    _y.push_back(y);
    _z.push_back(z);

    // fprintf(stderr, "{%f, %f, %f}\n", x, y, z);
    n ++;
  }

  assert(num_inclusions == _x.size());
  return true;
}
