#ifndef _DIY_EXT
#define _DIY_EXT

#include <diy/serialization.hpp>

static void bb2str(const diy::MemoryBuffer& bb, std::string& str)
{
  str.resize(bb.size());
  memcpy((char*)str.data(), bb.buffer.data(), bb.size());
}

static std::string bb2str(const diy::MemoryBuffer& bb)
{
  std::string str;
  bb2str(bb, str);
  return str;
}

#endif
