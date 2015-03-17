#include "VortexTransition.h"
#include <sstream>
#include <fstream>

VortexTransition::VortexTransition() :
  _num_global_vortices(0), 
  _ts(0), _tl(0)
{
}

VortexTransition::~VortexTransition()
{
}

void VortexTransition::LoadFromFile(const std::string& dataname, int ts, int tl)
{
  _ts = ts;
  _tl = tl;

  for (int i=ts; i<ts+tl-1; i++) {
    std::stringstream ss;
    ss << dataname << ".match." << i << "." << i+1;

    VortexTransitionMatrix tm;
    if (tm.LoadFromFile(ss.str()))
      insert(std::make_pair(i, tm));
    else 
      fprintf(stderr, "cannot open file %s\n", ss.str().c_str());
  }
}

void VortexTransition::SaveToDotFile(const std::string& filename)
{
  using namespace std;
  ofstream ofs(filename);
  if (!ofs.is_open()) return;

  ofs << "digraph {" << endl;
  ofs << "ratio = compress;" << endl;
  ofs << "rankdir = LR;" << endl;
  ofs << "ranksep =\"1.0 equally\";" << endl;
  ofs << "node [shape=circle];" << endl;
  for (int t=_ts; t<_ts+_tl-1; t++) {
    VortexTransitionMatrix tm = at(t);
    for (int i=0; i<tm.n0(); i++) {
      for (int j=0; j<tm.n1(); j++) {
        int weight = 1;
        if (tm.rowsum(i) == 1 && tm.colsum(j) == 1) weight = 1000;

        if (tm(i, j)) {
          ofs << t << "." << i << "->" 
              << t+1 << "." << j << " [weight = " << weight << "];" << endl;
        }
      }
    }
    
    ofs << "{ rank=same; ";
    for (int i=0; i<tm.n0(); i++) {
      if (i<tm.n0()-1) ofs << t << "." << i << ", ";
      else ofs << t << "." << i << " }" << endl;
    }
  }
  ofs << "}" << endl;
  ofs.close();
}

VortexTransitionMatrix VortexTransition::at(int i) const
{
  std::map<int, VortexTransitionMatrix>::const_iterator it = find(i);
  if (it != end())
    return it->second;
  else 
    return VortexTransitionMatrix();
}
