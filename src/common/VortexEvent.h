#ifndef _VORTEX_EVENT_H
#define _VORTEX_EVENT_H

#include <vector>

enum {
  VORTEX_EVENT_BIRTH = 0,
  VORTEX_EVENT_DEATH = 1,
  VORTEX_EVENT_MERGE = 2,
  VORTEX_EVENT_SPLIT = 3,
  VORTEX_EVENT_RECOMBINATION = 4,
  VORTEX_EVENT_ANNIHILATION = 5
};

struct VortexEvent {
  int timestep;
  double time;
  int type;

  std::vector<int> vids, vids1; // vortex ids before and after
};

bool SerializeVortexEvents(const std::vector<VortexEvent>& events, std::string& buf);
bool UnserializeVortexEvents(std::vector<VortexEvent>& events, const std::string& buf);

bool SaveVortexEvents(const std::vector<VortexEvent>& events, const std::string& filename);
bool LoadVortexEvents(std::vector<VortexEvent>& events, const std::string& filename); 

#endif
