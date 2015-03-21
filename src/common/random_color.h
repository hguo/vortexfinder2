#ifndef _RANDOM_COLOR_H
#define _RANDOM_COLOR_H

#include <vector>
#include <string>

void generate_random_colors(int count, std::vector<unsigned char>& colors);

std::string color2str(unsigned char r, unsigned char g, unsigned char b);

#endif
