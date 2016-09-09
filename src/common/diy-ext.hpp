#ifndef _DIY_EXT
#define _DIY_EXT

#include <diy/serialization.hpp>

static void bb2str(const diy::MemoryBuffer& bb, std::string& str)
{
  str.resize(bb.size());
  memcpy((char*)str.data(), bb.buffer.data(), bb.size());
}

static void str2bb(const std::string& str, diy::MemoryBuffer& bb)
{
  bb.reset();
  bb.buffer.resize(str.size());
  memcpy((char*)bb.buffer.data(), str.data(), str.size());
}

template <typename T> void serialize(const T& obj, std::string& buf)
{
  diy::MemoryBuffer bb;
  diy::save(bb, obj);
  bb2str(bb, buf);
}

template <typename T> void unserialize(const std::string& buf, T& obj)
{
  diy::MemoryBuffer bb;
  str2bb(buf, bb);
  diy::load(bb, obj);
}

#endif
