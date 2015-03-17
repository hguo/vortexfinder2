#include "VortexTransition.h"
#include <sstream>
#include <fstream>
#include <set>
#include <cassert>

VortexTransition::VortexTransition() :
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
      _matrices.insert(std::make_pair(i, tm));
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
    const VortexTransitionMatrix &tm = _matrices[t];
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

VortexTransitionMatrix VortexTransition::Matrix(int i) const
{
  std::map<int, VortexTransitionMatrix>::const_iterator it = _matrices.find(i);
  if (it != _matrices.end())
    return it->second;
  else 
    return VortexTransitionMatrix();
}

void VortexTransition::AddMatrix(const VortexTransitionMatrix& m)
{
  if (!m.Valid()) return;
  int t = m.t0();
  _matrices[t] = m;
}

int VortexTransition::NewVortexSequence(int ts)
{
  VortexSequence vs; 
  vs.ts = ts;
  vs.tl = 1;
  _seqs.push_back(vs);
  return _seqs.size() - 1;
}

int VortexTransition::SequenceIdx(int t, int lid) const
{
  std::tuple<int, int> key = std::make_tuple(t, lid);
  std::map<std::tuple<int,int>,int>::const_iterator it = _seqmap.find(key);
  if (it == _seqmap.end())
    return -1;
  else 
    return it->second;
}

void VortexTransition::ConstructSequence()
{
  for (int i=_ts; i<_ts+_tl-1; i++) {
    // fprintf(stderr, "=====t=%d\n", i);

    VortexTransitionMatrix tm = Matrix(i);
    assert(tm.Valid());

    const int n0 = tm.n0(), n1 = tm.n1(), 
              n = n0 + n1;
    
    if (i == _ts) { // initial ids;
      for (int k=0; k<n0; k++) {
        int gid = NewVortexSequence(i);
        _seqmap[std::make_tuple(i, k)] = gid;
      }
    }
    
    std::set<int> unvisited; 
    for (int i=0; i<n; i++) 
      unvisited.insert(i);
 
    while (!unvisited.empty()) {
      std::set<int> lhs, rhs; 
      std::vector<int> Q;
      Q.push_back(*unvisited.begin());

      while (!Q.empty()) {
        int v = Q.back();
        Q.pop_back();
        unvisited.erase(v);
        if (v<n0) lhs.insert(v);
        else rhs.insert(v-n0);

        if (v<n0) { // left hand side
          for (int j=0; j<n1; j++) 
            if (tm(v, j)>0 && unvisited.find(j+n0) != unvisited.end())
              Q.push_back(j+n0);
        } else {
          for (int i=0; i<n0; i++) 
            if (tm(i, v-n0)>0 && unvisited.find(i) != unvisited.end())
              Q.push_back(i);
        }
      }

      if (lhs.size() == 1 && rhs.size() == 1) { // ordinary case
        int l = *lhs.begin(), r = *rhs.begin();
        int gid = _seqmap[std::make_tuple(i, l)];
        _seqmap[std::make_tuple(i+1, r)] = gid;
        _seqs[gid].tl ++;
      } else { // some events, need re-ID
        for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
          int r = *it;
          int gid = NewVortexSequence(i+1);
          _seqmap[std::make_tuple(i+1, r)] = gid;
        }

        // right links
        for (std::set<int>::iterator it=lhs.begin(); it!=lhs.end(); it++) {
          int l = *it;
          int gid = _seqmap[std::make_tuple(i, l)];
          for (std::set<int>::iterator it1=rhs.begin(); it1!=rhs.end(); it1++) {
            // _seqs[gid].links_right.push_back(_seqmap[std::make_tuple(i+1, *it1)]);
          }
        }
      }
    
#if 0
      if (lhs.size() == 0 && rhs.size() == 1) {
        fprintf(stderr, "birth\n");
      } else if (lhs.size() == 1 && rhs.size() == 0) {
        fprintf(stderr, "death\n"); 
      } else if (lhs.size() == 1 && rhs.size() == 2) { // TODO: keep an ID
        fprintf(stderr, "split\n"); 
      } else if (lhs.size() == 2 && rhs.size() == 1) { // TODO: keep an ID
        fprintf(stderr, "merge\n");
      } else if (lhs.size() > 1 && rhs.size() > 1) { 
        fprintf(stderr, "recombination\n");
      }
#endif

#if 0
      int cnt=0;
      fprintf(stderr, "lhs={");
      for (std::set<int>::iterator it=lhs.begin(); it!=lhs.end(); it++) {
        if (cnt<lhs.size()-1) fprintf(stderr, "%d, ", *it);
        else fprintf(stderr, "%d}, ", *it);
        cnt ++;
      }
      cnt=0;
      fprintf(stderr, "rhs={");
      for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
        if (cnt<rhs.size()-1) fprintf(stderr, "%d, ", *it);
        else fprintf(stderr, "%d}\n", *it);
        cnt ++;
      }
#endif
    }
  }

#if 0
  for (int i=0; i<size(); i++) 
    fprintf(stderr, "gid=%d, ts=%d, tl=%d\n",
        i, at(i).ts, at(i).tl);
#endif
}
