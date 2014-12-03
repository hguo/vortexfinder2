#ifndef _VORTEX_OBJECT_H
#define _VORTEX_OBJECT_H

#include <string>
#include <list>
#include <vector>
#include <sstream>

/* 
 * \class   VortexObject
 * \author  Hanqi Guo
 * \brief   Vortex objects
*/
template <typename T=double>
class VortexObject : public std::vector<std::list<T> >
{
public:
  VortexObject() {}
  ~VortexObject() {}

  // FIXME: use production libraries like protobuf
  void Serialize(std::string& str) const {
    std::ostringstream stream; 
    stream << this->size() << "\t\n";  
    for (typename std::vector<std::list<T> >::const_iterator it = this->begin(); it != this->end(); it ++) 
      stream << it->size() << "\t";
    stream << "\n"; 
    for (typename std::vector<std::list<T> >::const_iterator it = this->begin(); it != this->end(); it ++) { 
      for (typename std::list<T>::const_iterator it1 = it->begin(); it1 != it->end(); it1 ++) 
        stream << *it1 << "\t"; 
      stream << "\n";
    }
    str = stream.str(); 
  }

  bool Unserialize(const std::string& str) {
    std::istringstream stream(str); 
    this->clear();

    size_t n; 
    stream >> n;
    this->resize(n); 

    std::vector<size_t> n_vertices(n);  
    for (size_t i=0; i<n; i++) {
      stream >> n_vertices[i];
      // fprintf(stderr, "n=%d, n_vertices=%d\n", n, n_vertices[i]); 
    }

    for (size_t i=0; i<n; i++) 
      for (size_t j=0; j<n_vertices[i]; j++) {
        T number; 
        stream >> number;
        this->at(i).push_back(number); 
      }

    return true; 
  }
}; 

#endif
