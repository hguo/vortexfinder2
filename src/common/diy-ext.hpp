#ifndef _DIY_EXT
#define _DIY_EXT

#include <diy/serialization.hpp>
#include <diy/storage.hpp>
#include <cassert>

namespace diy {
  struct StringBuffer : public BinaryBuffer {
    std::string &str;
    size_t pos; 

    explicit StringBuffer(std::string& str_, size_t pos_=0) : str(str_), pos(pos_) {}
    void clear() {str.clear(); pos = 0;}
    void reset() {pos = 0;}

    inline void save_binary(const char *x, size_t count) {
      if (pos + count > str.size()) str.resize(pos + count);
      memcpy((char*)(str.data()+pos), x, count);
      pos += count;
    }

    inline void load_binary(char *x, size_t count) {
      memcpy(x, str.data()+pos, count);
      pos += count;
    }

    inline void load_binary_back(char *x, size_t count) {
      memcpy(x, str.data()+str.size()-count, count);
    }
  };

  //////////
  template <typename T> void serialize(const T& obj, std::string& buf)
  {
    diy::StringBuffer bb(buf);
    diy::save(bb, obj);
  }

  template <typename T> void serializeToFile(const T& obj, const std::string& filename)
  {
    FILE *fp = fopen(filename.c_str(), "rb");
    assert(fp);
    diy::detail::FileBuffer bb(fp);
    diy::save(bb, obj);
    fclose(fp);
  }

  template <typename T> void unserialize(std::string& buf, T& obj)
  {
    diy::StringBuffer bb(buf);
    diy::load(bb, obj);
    buf.clear();
  }

  template <typename T> void unserializeFromFile(const std::string& filename, T& obj)
  {
    FILE *fp = fopen(filename.c_str(), "rb");
    assert(fp);
    diy::detail::FileBuffer bb(fp);
    diy::load(bb, obj);
    fclose(fp);
  }
}

#endif
