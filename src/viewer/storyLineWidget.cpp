#include <QMouseEvent>
#include <QKeyEvent>
#include <QFileDialog>
#include <QDebug>
#include <fstream>
#include "def.h"
#include "gl2ps/gl2ps.h"
#include "storyLineWidget.h"
#include "common/VortexTransition.h"
#include "common/Utils.hpp"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glu.h>
#include <GL/glut.h>
#endif

std::string color2str(unsigned char r, unsigned char g, unsigned char b)
{
  char buf[10];
  sprintf(buf, "#%02X%02X%02X", r, g, b);
  return std::string(buf);
}

CStorylineWidget::CStorylineWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget) :
  _vt(NULL)
{

}

CStorylineWidget::~CStorylineWidget()
{

}

void CStorylineWidget::parseLayout()
{
  using namespace std;
  ifstream ifs("dot.out");
  if (!ifs.is_open()) return;

  string str, label, type, shape, color, fillcolor;
  int id; 
  float x, y, w, h;

  ifs >> str >> id >> _layout_width >> _layout_height;
  while (1) {
    ifs >> str;
    if (str == "node") {
      ifs >> id >> x >> y >> w >> h >> label >> type >> shape >> color >> fillcolor;
      const int t = id / _vt->MaxNVorticesPerFrame(), 
                k = id % _vt->MaxNVorticesPerFrame();
      const QPair<int, int> key(t, k);
      _coords[key] = QVector2D(x, y);
      // fprintf(stderr, "t=%d, k=%d, x=%f, y=%f\n", t, k, x, y);
    } else 
      break;
  }

  ifs.close();
}

void CStorylineWidget::saveLayoutToJs()
{
  using namespace std;
  ofstream ofs("dot.js");
  if (!ofs.is_open()) return;

  ofs << "var dataset = [" << endl;

  const std::vector<struct VortexSequence> seqs = _vt->Sequences();
  for (int i=0; i<seqs.size(); i++) {
    const struct VortexSequence& seq = seqs[i];
    if (seq.lids.size() == 0) continue; // FIXME

    ofs << "\t[ ";
    for (int j=0; j<seq.lids.size(); j++) {
      const int t = seq.ts + j; 
      const int k = seq.lids[j];
      const QPair<int, int> key(t, k);
      QVector2D v = _coords[key];
      if (j<seq.lids.size()-1) 
        ofs << "[" << t << ", " << v.y() << "], ";
      else 
        ofs << "[" << t << ", " << v.y() << "]";
    }
    ofs << " ]," << endl;
  }

  ofs << "];" << endl;
  ofs.close();
}

void CStorylineWidget::saveLayoutToSVG(int w, int h)
{
  using namespace std;
  ofstream ofs("dot.svg");
  if (!ofs.is_open()) return;

  // ofs << "<svg width='" << _vt->tl() << "' height='" << _layout_height << "'>" << endl;
  ofs << "<svg width='" << w << "' height='" << h << "'>" << endl;

  const std::vector<struct VortexSequence> seqs = _vt->Sequences();
  std::map<int, VortexTransitionMatrix>& matrices = _vt->Matrices();
 
  // links
  for (int i=0; i<seqs.size(); i++) {
    int t = seqs[i].ts + seqs[i].tl - 1;
    if (t>=_vt->ts() + _vt->tl() - 1) continue; 
    int lhs_lid = seqs[i].lids.back();
    for (int k=0; k<matrices[t].n1(); k++) {
      if (matrices[t](lhs_lid, k)) {
        int rhs_lid = k;
        QVector2D v1 = _coords[qMakePair(t, lhs_lid)], 
                  v2 = _coords[qMakePair(t+1, rhs_lid)];
        ofs << "<line style='fill:none;stroke:grey;stroke-width:0.25' "
            << "x1='" << ((float)t-_vt->ts())/_vt->tl()*(w-1) << "' " 
            << "y1='" << v1.y()/_layout_height*h << "' "
            << "x2='" << ((float)(t+1)-_vt->ts())/_vt->tl()*(w-1) << "' "
            << "y2='" << v2.y()/_layout_height*h << "' />" << endl;
      }
    }
  }

  // vortices 
  for (int i=0; i<seqs.size(); i++) {
    const struct VortexSequence& seq = seqs[i];
    // if (seq.lids.size() == 0) continue; // FIXME
    const std::string color = color2str(seq.r, seq.g, seq.b);

    ofs << "<polyline style='fill:none;stroke:" << color << ";stroke-width:1' points='";
    for (int j=0; j<seq.lids.size(); j++) {
      const int t = seq.ts + j; 
      const int k = seq.lids[j];
      const QPair<int, int> key(t, k);
      QVector2D v = _coords[key];
      ofs << ((float)t-_vt->ts())/_vt->tl()*(w-1) << "," << v.y()/_layout_height*h << " ";
    }
    ofs << "'/>" << endl;
  }

  ofs << "</svg>" << endl;
  fprintf(stderr, "saved to svg file.\n");
  exit(1);

  ofs.close();
}

void CStorylineWidget::SetVortexTrasition(VortexTransition *vt)
{
  _vt = vt;

  vt->SaveToDotFile("dot");
  int succ = system("dot -Tplain dot > dot.out");
  if (succ != 0) {
    fprintf(stderr, "FATAL: graphviz not available. exiting.\n");
    exit(1);
  }

  parseLayout();
  // saveLayoutToJs();
  saveLayoutToSVG(1000, 50);
}

void CStorylineWidget::initializeGL()
{
  glEnable(GL_MULTISAMPLE);

  GLint bufs, samples; 
  glGetIntegerv(GL_SAMPLE_BUFFERS, &bufs); 
  glGetIntegerv(GL_SAMPLES, &samples); 

  glEnable(GL_LINE_SMOOTH); 
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); 
  
  glEnable(GL_POLYGON_SMOOTH); 
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST); 

  glEnable(GL_BLEND); 
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

  glClearColor(1, 1, 1, 0);
}

void CStorylineWidget::resizeGL(int w, int h)
{
  const int margin = 10; // 10px
  _rect_chart = QRectF(margin, margin, w-2*margin, h-2*margin);
}

void CStorylineWidget::renderRect()
{
  glColor3f(0, 0, 0);
  glBegin(GL_LINE_LOOP);
  glVertex2f(_rect_chart.left(), _rect_chart.bottom());
  glVertex2f(_rect_chart.right(), _rect_chart.bottom());
  glVertex2f(_rect_chart.right(), _rect_chart.top());
  glVertex2f(_rect_chart.left(), _rect_chart.top());
  glEnd();
}

void CStorylineWidget::renderLines()
{
  const std::vector<struct VortexSequence> seqs = _vt->Sequences();
  
  if (_coords.empty()) return;

  glPushMatrix();

  glTranslatef(_rect_chart.x(), _rect_chart.y(), 0);
  glScalef(_rect_chart.width(), _rect_chart.height(), 1);
  glScalef(1/_layout_width, 1/_layout_height, 1);

  glColor3f(1, 0, 0);

  for (int i=0; i<seqs.size(); i++) {
    const struct VortexSequence& seq = seqs[i];
    glColor3ub(seq.r, seq.g, seq.b);

    glBegin(GL_LINE_STRIP);
    for (int j=0; j<seq.lids.size(); j++) {
      const int t = seq.ts + j; 
      const int k = seq.lids[j];
      const QPair<int, int> key(t, k);
      if (!_coords.contains(key)) qDebug() << key;
      QVector2D v = _coords[key];
      glVertex2f(v.x(), v.y());
    }
    glEnd();
  }

  glPopMatrix();
}

void CStorylineWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, width(), height());

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, width(), 0, height());
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // renderRect();
  renderLines();

  CHECK_GLERROR();
}

void CStorylineWidget::saveEps()
{
  QString filename = QFileDialog::getSaveFileName(this, "save to eps", "./", "*.eps"); 
  if (filename.isEmpty()) return; 

  FILE *fp = fopen(filename.toStdString().c_str(), "wb"); 
  if (!fp) return; 
  fprintf(stderr, "saving eps %s\n", filename.toStdString().c_str()); 

  int state = GL2PS_OVERFLOW; 
  int bufsize = 0; 

  while (state == GL2PS_OVERFLOW) {
    bufsize += 1024*1024; 
    gl2psBeginPage("test", "test", NULL, GL2PS_EPS, GL2PS_SIMPLE_SORT, 
                   GL2PS_DRAW_BACKGROUND | GL2PS_USE_CURRENT_VIEWPORT, 
                   GL_RGBA, 0, NULL, 0, 0, 0, bufsize, fp, filename.toStdString().c_str()); 
    paintGL(); 
    state = gl2psEndPage(); 
  }
  fclose(fp); 
}

void CStorylineWidget::keyPressEvent(QKeyEvent *e)
{
  switch (e->key()) {
  case Qt::Key_E:
    saveEps(); 
    break;

  default: break;
  }
}
