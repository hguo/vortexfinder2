#include <QMouseEvent>
#include <QDebug>
#include <fstream>
#include "def.h"
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

void CStorylineWidget::SetVortexTrasition(const VortexTransition *vt)
{
  _vt = vt;

  vt->SaveToDotFile("dot");
  int succ = system("dot -Tplain dot > dot.out");
  if (succ != 0) {
    fprintf(stderr, "FATAL: graphviz not available. exiting.\n");
    exit(1);
  }

  parseLayout();
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
  const int nc = 6;
  const GLubyte c[nc][3] = {
    {0, 0, 255},
    {0, 255, 0},
    {0, 255, 255},
    {255, 0, 0},
    {255, 0, 255},
    {255, 255, 0}};
  const std::vector<struct VortexSequence> seqs = _vt->Sequences();
  
  if (_coords.empty()) return;

  glPushMatrix();

  glTranslatef(_rect_chart.x(), _rect_chart.y(), 0);
  glScalef(_rect_chart.width(), _rect_chart.height(), 1);
  glScalef(1/_layout_width, 1/_layout_height, 1);

  glColor3f(1, 0, 0);

  for (int i=0; i<seqs.size(); i++) {
    const struct VortexSequence& seq = seqs[i];
    const int cid = i % nc;
    glColor3ub(c[cid][0], c[cid][1], c[cid][2]);

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
