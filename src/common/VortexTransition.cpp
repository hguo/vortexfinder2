#include "VortexTransition.h"
#include <sstream>
#include <fstream>
#include <set>
#include <cassert>
#include "random_color.h"

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
    if (tm.LoadFromFile(ss.str())) {
      _matrices.insert(std::make_pair(i, tm));
      _nvortices_per_frame[i] = tm.n0(); 
      _nvortices_per_frame[i+1] = tm.n1();
    }
    else 
      fprintf(stderr, "cannot open file %s\n", ss.str().c_str());
  }

  _max_nvortices_per_frame = 0;
  for (std::map<int, int>::iterator it = _nvortices_per_frame.begin(); it != _nvortices_per_frame.end(); it ++) {
    _max_nvortices_per_frame = std::max(_max_nvortices_per_frame, it->second);
  }
  // fprintf(stderr, "max_nvortices_per_frame=%d\n", _max_nvortices_per_frame);
}

#define HUMAN_READABLE 0
std::string VortexTransition::NodeToString(int i, int j) const
{
  using namespace std;
  stringstream ss;
#if HUMAN_READABLE
  ss << i << "." << j;
#else
  ss << i*_max_nvortices_per_frame + j;
#endif
  return ss.str();
}

void VortexTransition::SaveToDotFile(const std::string& filename) const
{
  using namespace std;
  ofstream ofs(filename);
  if (!ofs.is_open()) return;

  ofs << "digraph {" << endl;
  ofs << "ratio = compress;" << endl;
  ofs << "rankdir = LR;" << endl;
  ofs << "ranksep =\"1.0 equally\";" << endl;
  ofs << "node [shape=circle];" << endl;
  // ofs << "node [shape=point];" << endl;
#if 1
  for (int t=_ts; t<_ts+_tl-1; t++) {
    const VortexTransitionMatrix &tm = Matrix(t); 
    for (int i=0; i<tm.n0(); i++) {
      for (int j=0; j<tm.n1(); j++) {
        int weight = 1;
        if (tm.rowsum(i) == 1 && tm.colsum(j) == 1) weight = 1000;

        if (tm(i, j)) {
          ofs << NodeToString(t, i) << "->" 
              << NodeToString(t+1, j)
              << " [weight = " << weight << "];" << endl;
        }
      }
    }
    
    ofs << "{ rank=same; ";
    for (int i=0; i<tm.n0(); i++) {
      if (i<tm.n0()-1) ofs << NodeToString(t, i) << ", ";
      else ofs << NodeToString(t, i) << "}" << endl;
    }
  }
#else
  // iteration over sequences
  for (int i=0; i<_seqs.size(); i++) {
    const VortexSequence &seq = _seqs[i];
    for (int k=0; k<seq.lids.size(); k++) {
      const int t = seq.ts + k;
      const int weight = seq.tl;
      if (k<seq.lids.size()-1) 
        ofs << NodeToString(t, seq.lids[k]) << "->";
      else 
        ofs << NodeToString(t, seq.lids[k]) 
            << " [weight = " << weight << "];" << endl;
    }
  }
  // ranks
  for (int t=_ts; t<_ts+_tl-1; t++) {
    std::map<int, int>::const_iterator it = _nvortices_per_frame.find(t); 
    const int n = it->second;
    ofs << "{ rank=same; ";
    for (int i=0; i<n; i++) {
      if (i<n-1) ofs << NodeToString(t, i) << ", ";
      else ofs << NodeToString(t, i) << " }" << endl;
    }
  }
  // subgraphs
  for (int i=0; i<_events.size(); i++) {
    int t = _events[i].t;
    ofs << "subgraph {";
    for (std::set<int>::iterator it0 = _events[i].lhs.begin(); it0 != _events[i].lhs.end(); it0 ++) 
      for (std::set<int>::iterator it1 = _events[i].rhs.begin(); it1 != _events[i].rhs.end(); it1 ++) {
        ofs << NodeToString(t, *it0) << "->" 
            << NodeToString(t+1, *it1) << ";" << endl;
      }
    ofs << "};" << endl;
  }
#endif
  // node colors
  for (int t=_ts; t<_ts+_tl; t++) {
    std::map<int, int>::const_iterator it = _nvortices_per_frame.find(t); 
    const int n = it->second;
    for (int k=0; k<n; k++) {
      const int nc = 6;
      int vid = SequenceIdx(t, k);
      int c = vid % nc;
      std::string color;
      
      if (c == 0) color = "blue";
      else if (c == 1) color = "green";
      else if (c == 2) color = "cyan";
      else if (c == 3) color = "red";
      else if (c == 4) color = "purple";
      else if (c == 5) color = "yellow";

#if HUMAN_READABLE
      ofs << t << "." << k 
          << " [style=filled, fillcolor=" << color << "];" << endl;
#else
      ofs << t*_max_nvortices_per_frame+k
          << " [style=filled, fillcolor=" << color << "];" << endl;
#endif
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
  vs.tl = 0;
  // vs.lhs_event = vs.rhs_event = VORTEX_EVENT_DUMMY;
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

void VortexTransition::SequenceColor(int gid, unsigned char &r, unsigned char &g, unsigned char &b) const
{
  r = _seqs[gid].r;
  g = _seqs[gid].g;
  b = _seqs[gid].b;
}

void VortexTransition::ConstructSequence()
{
  for (int i=_ts; i<_ts+_tl-1; i++) {
    // fprintf(stderr, "=====t=%d\n", i);

    VortexTransitionMatrix tm = Matrix(i);
    assert(tm.Valid());

    if (i == _ts) { // initial
      for (int k=0; k<tm.n0(); k++) {
        int gid = NewVortexSequence(i);
        _seqs[gid].tl ++;
        _seqs[gid].lids.push_back(k);
        _seqmap[std::make_tuple(i, k)] = gid;
      }
    }

    for (int k=0; k<tm.NModules(); k++) {
      int event;
      std::set<int> lhs, rhs;
      tm.GetModule(k, lhs, rhs, event);

      if (lhs.size() == 1 && rhs.size() == 1) { // ordinary case
        int l = *lhs.begin(), r = *rhs.begin();
        int gid = _seqmap[std::make_tuple(i, l)];
        _seqs[gid].tl ++;
        _seqs[gid].lids.push_back(r);
        _seqmap[std::make_tuple(i+1, r)] = gid;
      } else { // some events, need re-ID
        for (std::set<int>::iterator it=rhs.begin(); it!=rhs.end(); it++) {
          int r = *it; 
          int gid = NewVortexSequence(i+1);
          _seqs[gid].tl ++;
          _seqs[gid].lids.push_back(r);
          _seqmap[std::make_tuple(i+1, r)] = gid;
        }
      }

      // build events
      if (event >= VORTEX_EVENT_MERGE) {
        VortexEvent e;
        e.t = i;
        e.event = event;
        e.lhs = lhs;
        e.rhs = rhs;
        _events.push_back(e);
      }
    }
  }

  RandomColorSchemes();

#if 0
  for (int i=0; i<_events.size(); i++) 
    fprintf(stderr, "e=%d, #l=%d, #r=%d\n", _events[i].event, _events[i].lhs.size(), _events[i].rhs.size());
#endif
}


#if 0
      if (event >= VORTEX_EVENT_MERGE) { // FIXME: other events
        for (int u=0; u<lhs.size(); u++) {
          const int lgid = _seqmap[std::make_tuple(i, lhs[u])];
          _seqs[lgid].rhs_event = event;
          _seqs[lgid].rhs_gids = rhs;
        }
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

void VortexTransition::RandomColorSchemes()
{
  std::vector<unsigned char> colors;
  generate_random_colors(_seqs.size(), colors);

  for (int i=0; i<_seqs.size(); i++) {
    _seqs[i].r = colors[i*3];
    _seqs[i].g = colors[i*3+1];
    _seqs[i].b = colors[i*3+2];
  }
}
