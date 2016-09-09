#ifndef _DIY_EXT
#define _DIY_EXT

#include <diy/serialization.hpp>
#include <diy/storage.hpp>
#include <cassert>

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

template <typename T> void serializeToFile(const T& obj, const std::string& filename)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  assert(fp);
  diy::detail::FileBuffer bb(fp);
  diy::save(bb, obj);
  fclose(fp);
}

template <typename T> void unserialize(const std::string& buf, T& obj)
{
  diy::MemoryBuffer bb;
  str2bb(buf, bb);
  diy::load(bb, obj);
}

template <typename T> void unserializeFromFile(const std::string& filename, T& obj)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  assert(fp);
  diy::detail::FileBuffer bb(fp);
  diy::load(bb, obj);
  fclose(fp);
}

#endif
