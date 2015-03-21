#include "random_color.h"
#include "zcolor.h"
#include <algorithm>

void generate_random_colors(int count, std::vector<unsigned char>& colors)
{
  const double saturation = 0.7, 
               intensity = 0.5;

  std::vector<double> hues(count);
  for (int i=0; i<count; i++) 
    hues[i] = (double)i/(count-1);
  
  std::random_shuffle(hues.begin(), hues.end());

  for (int i=0; i<count; i++) {
    ZHSI hsi;
    hsi.hue = hues[i];
    hsi.saturation = saturation;
    hsi.intensity = intensity;

    ZRGB rgb;
    ZColor::HSI2RGB(&hsi, &rgb);

    colors.push_back(rgb.red * 255);
    colors.push_back(rgb.green * 255);
    colors.push_back(rgb.blue * 255);
  }
}

void generate_colors(int count, std::vector<unsigned char>& colors)
{
  const double saturation = 0.7, 
               intensity = 0.5;

  std::vector<double> hues(count);
  for (int i=0; i<count; i++) 
    hues[i] = (double)i/(count-1);
  
  for (int i=0; i<count; i++) {
    ZHSI hsi;
    hsi.hue = hues[i];
    hsi.saturation = saturation;
    hsi.intensity = intensity;

    ZRGB rgb;
    ZColor::HSI2RGB(&hsi, &rgb);

    colors.push_back(rgb.red * 255);
    colors.push_back(rgb.green * 255);
    colors.push_back(rgb.blue * 255);
  }
}

