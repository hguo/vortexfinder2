#include <QMouseEvent>
#include <QDebug>
#include <fstream>
#include <iostream>
#include "widget.h"
#include "common/VortexLine.h"
#include "common/FieldLine.h"
#include "common/Utils.hpp"

#ifdef __APPLE__
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glu.h>
#include <GL/glut.h>
#endif

using namespace std; 

CGLWidget::CGLWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget)
  : QGLWidget(fmt, parent, sharedWidget), 
    _fovy(30.f), _znear(0.1f), _zfar(10.f), 
    _eye(0, 0, 2.5), _center(0, 0, 0), _up(0, 1, 0), 
    _vortex_render_mode(0), 
    _enable_inclusions(false)
{
}

CGLWidget::~CGLWidget()
{
}

void CGLWidget::SetDataName(const std::string& dataname)
{
  _dataname = dataname;
}

void CGLWidget::LoadTimeStep(int t)
{
  _timestep = t;
  
  stringstream ss;
  ss << _dataname << ".vlines." << t;

  Clear();
  LoadVortexLines(ss.str());
}

void CGLWidget::mousePressEvent(QMouseEvent* e)
{
  _trackball.mouse_rotate(e->x(), e->y()); 
}

void CGLWidget::mouseMoveEvent(QMouseEvent* e)
{
  _trackball.motion_rotate(e->x(), e->y()); 
  updateGL(); 
}

void CGLWidget::keyPressEvent(QKeyEvent* e)
{
  switch (e->key()) {
  case Qt::Key_Comma: 
    LoadTimeStep(_timestep - 1); 
    updateGL();
    break;

  case Qt::Key_Period: 
    LoadTimeStep(_timestep + 1);
    updateGL();
    break;

  case Qt::Key_T:
    _vortex_render_mode = 0;
    updateGL();
    break;

  case Qt::Key_L:
    _vortex_render_mode = 1;
    updateGL();
    break;

  case Qt::Key_I:
    _enable_inclusions = !_enable_inclusions;
    updateGL();
    break;

  default: break; 
  }
}

void CGLWidget::wheelEvent(QWheelEvent* e)
{
  _trackball.wheel(e->delta());
  updateGL(); 
}

void CGLWidget::initializeGL()
{
  _trackball.init();

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
  
  // initialze light for tubes
  {
    GLfloat ambient[]  = {0.1, 0.1, 0.1}, 
            diffuse[]  = {0.5, 0.5, 0.5}, 
            specular[] = {0.8, 0.8, 0.8}; 
    GLfloat dir[] = {0, 0, -1}; 
    GLfloat shiness = 100; 

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient); 
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse); 
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular); 
    // glLightfv(GL_LIGHT0, GL_POSITION, pos); 
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir); 
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE); 

    glEnable(GL_NORMALIZE); 
    glEnable(GL_COLOR_MATERIAL); 
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular); 
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shiness); 
  }

  CHECK_GLERROR(); 
}

void CGLWidget::resizeGL(int w, int h)
{
  _trackball.reshape(w, h); 
  glViewport(0, 0, w, h);
  
  CHECK_GLERROR(); 
}

void CGLWidget::renderFieldLines()
{
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 
  glVertexPointer(3, GL_FLOAT, 0, f_line_vertices.data()); 
  glColorPointer(4, GL_FLOAT, 4*sizeof(GLfloat), f_line_colors.data());
  
  glMultiDrawArrays(GL_LINE_STRIP, f_line_indices.data(), f_line_vert_count.data(), f_line_vert_count.size());

  glDisableClientState(GL_COLOR_ARRAY); 
  glDisableClientState(GL_VERTEX_ARRAY); 
}

void CGLWidget::renderVortexIds()
{
  QFont ft;
  ft.setPointSize(36);

  glColor3f(0, 0, 0);
  glDisable(GL_DEPTH_TEST);

  QString s0 = QString("timestep=%1").arg(_timestep);
  renderText(20, 60, s0, ft);

  ft.setPointSize(24);
  glColor3f(0, 0, 0);
  for (int i=0; i<_vids.size(); i++) {
    int id = _vids[i];
    QVector3D v = _vids_coord[i];
    QString s = QString("%1").arg(id);
#if 0
    glPushMatrix();
    glTranslatef(v.x(), v.y(), v.z());
    glutSolidSphere(0.5, 20, 20);
    glPopMatrix();
#endif

    renderText(v.x(), v.y(), v.z(), s, ft);
  }
}

typedef struct {
  QVector3D p;
  QColor c;
  float depth;
} inclusion_t;

static bool compare_inclusions(const inclusion_t& i0, const inclusion_t& i1)
{
  return i0.depth < i1.depth;
}

void CGLWidget::renderInclusions()
{
  const int n = 10;
  const float radius = 5.f;
  const GLubyte alpha = 128;
  const QVector3D p[n] = {
    {56.156670, 51.160450, 3.819186},
    {62.730570, 43.044800, 8.598517},
    {47.607200, 53.324570, 11.099090},
    {26.116400, 30.941740, 3.956855},
    {86.089940, 50.946700, 6.626538},
    {93.094290, 56.579140, 11.743990},
    {83.132140, 25.316290, 9.010600},
    {12.312030, 50.503210, 7.045643},
    {38.015730, 12.054860, 11.574300},
    {85.341200, 36.842770, 6.001254}};
  const GLubyte c[n][3] = {
    {230, 13, 13},
    {8, 138, 138},
    {230, 111, 13},
    {53, 195, 53},
    {0, 121, 0},
    {151, 68, 0},
    {0, 86, 0}, 
    {108, 49, 0},
    {151, 0, 0},
    {243, 146, 66}};
  inclusion_t inclusions[n];

  for (int i=0; i<n; i++) {
    inclusions[i].p = p[i];
    QVector3D v = _mvmatrix * p[i];
    inclusions[i].depth = v.z();
    inclusions[i].c = QColor(c[i][0], c[i][1], c[i][2]);
  }

  qSort(inclusions, inclusions+n-1, compare_inclusions);
  
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 
  glEnable(GL_CULL_FACE);

  glPushMatrix();
  glTranslatef(-64, -32, -8); // FIXME: hard code
  for (int i=0; i<n; i++) {
    glColor4ub(inclusions[i].c.red(), inclusions[i].c.green(), inclusions[i].c.blue(), 128);
    glPushMatrix();
    glTranslatef(inclusions[i].p.x(), inclusions[i].p.y(), inclusions[i].p.z());
    glutSolidSphere(radius, 20, 20);
    glPopMatrix();
  }
  glPopMatrix();

  glPopAttrib();
}

void CGLWidget::renderVortexArrows()
{
  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 

  for (int i=0; i<_cones_pos.size(); i++) {
    QColor c = _cones_color[i];
    QVector3D p = _cones_pos[i];
    QVector3D d = _cones_dir[i];
    QVector3D z(0, 0, 1);
    QVector3D a = QVector3D::crossProduct(z, d);
    float omega = acos(QVector3D::dotProduct(z, d)) * 180 / M_PI;

    glColor3ub(c.red(), c.green(), c.blue());
    glPushMatrix();
    glTranslatef(p.x(), p.y(), p.z());
    glRotatef(omega, a.x(), a.y(), a.z());
    glutSolidCone(1, 3, 12, 4); 
    glPopMatrix();
  }

  glPopAttrib();
}

void CGLWidget::renderVortexLines()
{
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 
  glVertexPointer(3, GL_FLOAT, 0, v_line_vertices.data()); 
  glColorPointer(3, GL_UNSIGNED_BYTE, 3*sizeof(GLubyte), v_line_colors.data());
  
  glMultiDrawArrays(GL_LINE_STRIP, v_line_indices.data(), v_line_vert_count.data(), v_line_vert_count.size());
  // glMultiDrawArrays(GL_POINTS, v_line_indices.data(), v_line_vert_count.data(), v_line_vert_count.size());

  glDisableClientState(GL_COLOR_ARRAY); 
  glDisableClientState(GL_VERTEX_ARRAY); 
}

void CGLWidget::renderVortexTubes()
{
  glPushAttrib(GL_ENABLE_BIT);

  glEnable(GL_DEPTH_TEST); 
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 

  glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT); 
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_NORMAL_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 

  glVertexPointer(3, GL_FLOAT, 0, vortex_tube_vertices.data()); 
  glNormalPointer(GL_FLOAT, 0, vortex_tube_normals.data()); 
  glColorPointer(3, GL_UNSIGNED_BYTE, 0, vortex_tube_colors.data()); 
  glDrawElements(GL_TRIANGLES, vortex_tube_indices_vertices.size(), GL_UNSIGNED_INT, vortex_tube_indices_vertices.data()); 

  glPopClientAttrib(); 

  glPopAttrib();
}

void CGLWidget::paintGL()
{
  glClearColor(1, 1, 1, 0); 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

  _projmatrix.setToIdentity(); 
  _projmatrix.perspective(_fovy, (float)width()/height(), _znear, _zfar); 
  _mvmatrix.setToIdentity();
  _mvmatrix.lookAt(_eye, _center, _up);
  _mvmatrix.rotate(_trackball.getRotation());
  _mvmatrix.scale(_trackball.getScale());
  _mvmatrix.scale(0.02);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glLoadMatrixd(_projmatrix.data()); 
  glMatrixMode(GL_MODELVIEW); 
  glLoadIdentity(); 
  glLoadMatrixd(_mvmatrix.data()); 

#if 0
  glColor3f(0.f, 0.f, 0.f);
  glPushMatrix();
  glScalef(128.f, 128.f, 64.f);
  glutWireCube(1.0);
  glPopMatrix();
#endif

  if (_vortex_render_mode == 0)
    renderVortexTubes();
  else 
    renderVortexLines();

  if (_enable_inclusions)
    renderInclusions();

  // renderVortexArrows();
  renderVortexIds();

  renderFieldLines();

  CHECK_GLERROR(); 
}

void CGLWidget::LoadFieldLines(const std::string& filename)
{
  std::vector<FieldLine> fieldlines;
  ReadFieldLines(filename, fieldlines);

  float c[4] = {0, 0, 0, 1}; // color;
  for (int i=0; i<fieldlines.size(); i++) {
    int vertCount = fieldlines[i].size()/3;  
   
    for (int j=0; j<fieldlines[i].size(); j++) 
      f_line_vertices.push_back(fieldlines[i][j]); 
    
    for (int l=0; l<vertCount; l++)  
      for (int m=0; m<4; m++) 
        f_line_colors.push_back(c[m]); 

    f_line_vert_count.push_back(vertCount); 
  }
  
  int cnt = 0; 
  for (int i=0; i<f_line_vert_count.size(); i++) {
    f_line_indices.push_back(cnt); 
    cnt += f_line_vert_count[i]; 
  }
}

void CGLWidget::Clear()
{
  v_line_vertices.clear();
  v_line_colors.clear();
  v_line_vert_count.clear();
  v_line_indices.clear();
  vortex_tube_vertices.clear();
  vortex_tube_normals.clear();
  vortex_tube_colors.clear();
  vortex_tube_indices_lines.clear();
  vortex_tube_indices_vertices.clear();

  f_line_vertices.clear();
  f_line_colors.clear();
  f_line_vert_count.clear();
  f_line_indices.clear();

  _vids.clear();
  _vids_coord.clear();

  _cones_pos.clear();
  _cones_dir.clear();
  _cones_color.clear();
}

void CGLWidget::LoadVortexLines(const std::string& filename)
{
  std::string info_bytes;
  std::vector<VortexLine> vortex_liness;
  if (!::LoadVortexLines(vortex_liness, info_bytes, filename))
    return;

  fprintf(stderr, "Loaded vortex line file from %s\n", filename.c_str());

  if (info_bytes.length()>0) 
    _data_info.ParseFromString(info_bytes);

  const double O[3] = {_data_info.ox(), _data_info.oy(), _data_info.oz()},
               L[3] = {_data_info.lx(), _data_info.ly(), _data_info.lz()};

  const int nc = 6;
  const GLubyte c[nc][3] = {
    {0, 0, 255},
    {0, 255, 0},
    {0, 255, 255},
    {255, 0, 0},
    {255, 0, 255},
    {255, 255, 0}};

  int vertCount = 0; 
  for (int k=0; k<vortex_liness.size(); k++) { //iterator over lines
    // fprintf(stderr, "line %d: id=%d, len=%lu\n", k, vortex_liness[k].id, vortex_liness[k].size());
    if (vortex_liness[k].size()>=3) {
      _vids.push_back(vortex_liness[k].id);
      QVector3D pt(*(vortex_liness[k].begin()), 
                   *(vortex_liness[k].begin()+1),
                   *(vortex_liness[k].begin()+2));
      _vids_coord.push_back(pt);
    }

    int ci = vortex_liness[k].id % nc; // color index

    if (vortex_liness[k].is_bezier) { // TODO: make it more graceful..
      VortexLine& vl = vortex_liness[k];
      const int span = 6;

      for (int i=4*span; i<vl.size()/3; i+=4*span) {
        QVector3D p(
            fmod1(vl[i*3]-O[0], L[0]) + O[0], 
            fmod1(vl[i*3+1]-O[1], L[1]) + O[1], 
            fmod1(vl[i*3+2]-O[2], L[2]) + O[2]);
        QVector3D p0(vl[i*3], vl[i*3+1], vl[i*3+2]), 
                  p1(vl[i*3+3], vl[i*3+4], vl[i*3+5]);
        QVector3D d = (p1 - p0).normalized();
        QColor color(c[ci][0], c[ci][1], c[ci][2]);

        _cones_pos.push_back(p);
        _cones_dir.push_back(d);
        _cones_color.push_back(color);
      }

      vl.ToRegular(0.02);
      vl.Unflattern(O, L);
    }
    
    std::vector<double>::iterator it = vortex_liness[k].begin();
    QVector3D p0;
    for (int i=0; i<vortex_liness[k].size()/3; i++) {
      QVector3D p(*it, *(++it), *(++it));
      it ++;
      
      v_line_vertices.push_back(p.x()); 
      v_line_vertices.push_back(p.y()); 
      v_line_vertices.push_back(p.z()); 
      v_line_colors.push_back(c[ci][0]); 
      v_line_colors.push_back(c[ci][1]); 
      v_line_colors.push_back(c[ci][2]); 

      if (i>0 && (p-p0).length()>5) {
        v_line_vert_count.push_back(vertCount); 
        vertCount = 0;
      }
      p0 = p;
      
      vertCount ++;
    }

    if (vertCount != 0) {
      v_line_vert_count.push_back(vertCount); 
      vertCount = 0;
    }
  }
  
  int cnt = 0; 
  for (int i=0; i<v_line_vert_count.size(); i++) {
    v_line_indices.push_back(cnt); 
    cnt += v_line_vert_count[i]; 
    // fprintf(stderr, "break, vertCount=%d, lineIndices=%d\n", v_line_vert_count[i], v_line_indices[i]);
  }
  // v_line_indices.push_back(cnt); 

  updateVortexTubes(20, 0.5); 
}

void CGLWidget::LoadVortexLinesFromTextFile(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) return;

  const float c[3] = {1, 0, 0};

  std::string line;
  int vertCount = 0;
  getline(ifs, line);

  float x0, y0, z0, x, y, z;
  while (getline(ifs, line)) {
    if (line[0] == '#') {
      v_line_vert_count.push_back(vertCount); 
      vertCount = 0;
      continue;
    }

    std::stringstream ss(line);
    ss >> x >> y >> z;
        
    v_line_vertices.push_back(x); 
    v_line_vertices.push_back(y); 
    v_line_vertices.push_back(z);
    for (int m=0; m<3; m++) 
      v_line_colors.push_back(c[m]); 

    if (vertCount > 1) {
      float dist = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)); 
      if (dist > 3) {
        v_line_vert_count.push_back(vertCount); 
        vertCount = 0;
      }
    }
    x0 = x; y0 = y; z0 = z;

    vertCount ++;
  }

  if (vertCount != 0)
    v_line_vert_count.push_back(vertCount); 
  
  int cnt = 0; 
  for (int i=0; i<v_line_vert_count.size(); i++) {
    v_line_indices.push_back(cnt); 
    cnt += v_line_vert_count[i]; 
  }
  v_line_indices.push_back(cnt); 
    
  
  fprintf(stderr, "n=%ld\n", v_line_vertices.size()/3); 
  
  updateVortexTubes(20, 0.5); 
}

void CGLWidget::LoadInclusionsFromTextFile(const std::string& filename)
{

}

////////////////
void CGLWidget::updateVortexTubes(int nPatches, float radius) 
{
  vortex_tube_vertices.clear(); 
  vortex_tube_normals.clear(); 
  vortex_tube_colors.clear(); 
  vortex_tube_indices_lines.clear(); 
  vortex_tube_indices_vertices.clear(); 

  for (int i=0; i<v_line_vert_count.size(); i++) {
    if (v_line_vert_count[i] < 2) continue; 
     
    int first = v_line_indices[i]; 
    QVector3D N0; 
    for (int j=1; j<v_line_vert_count[i]; j++) {
      QVector3D P0 = QVector3D(v_line_vertices[(first+j-1)*3], v_line_vertices[(first+j-1)*3+1], v_line_vertices[(first+j-1)*3+2]); 
      QVector3D P  = QVector3D(v_line_vertices[(first+j)*3], v_line_vertices[(first+j)*3+1], v_line_vertices[(first+j)*3+2]);
      GLubyte color[3] = {v_line_colors[(first+j)*3], v_line_colors[(first+j)*3+1], v_line_colors[(first+j)*3+2]}; 

      QVector3D T = (P - P0).normalized(); 
      QVector3D N = QVector3D(-T.y(), T.x(), 0.0).normalized(); 
      QVector3D B = QVector3D::crossProduct(N, T); 

      // if (N.length() == 0 || isnan(N.length())) continue;

      if (j>1) {
        float n0 = QVector3D::dotProduct(N0, N); 
        float b0 = QVector3D::dotProduct(N0, B);
        QVector3D N1 = n0 * N + b0 * B;
        N = N1.normalized(); 
        B = QVector3D::crossProduct(N, T).normalized(); 
      }
      N0 = N;

      const int nIteration = (j==1)?2:1; 
      for (int k=0; k<nIteration; k++) {
        for (int p=0; p<nPatches; p++) {
          float angle = p * 2.f * M_PI / nPatches; 
          QVector3D normal = (N*cos(angle) + B*sin(angle)).normalized(); 
          QVector3D offset = normal * radius; 
          QVector3D coord; 

          if (k==0 && j==1) coord = P0 + offset; 
          else coord = P + offset; 

          vortex_tube_vertices.push_back(coord.x()); 
          vortex_tube_vertices.push_back(coord.y()); 
          vortex_tube_vertices.push_back(coord.z()); 
          // vortex_tube_vertices.push_back(1); 
          vortex_tube_normals.push_back(normal.x()); 
          vortex_tube_normals.push_back(normal.y()); 
          vortex_tube_normals.push_back(normal.z()); 
          vortex_tube_colors.push_back(color[0]); 
          vortex_tube_colors.push_back(color[1]); 
          vortex_tube_colors.push_back(color[2]);
          vortex_tube_indices_lines.push_back(j); 
        }
      }

      for (int p=0; p<nPatches; p++) {
        const int n = vortex_tube_vertices.size()/3; 
        const int pn = (p+1)%nPatches; 
        vortex_tube_indices_vertices.push_back(n-nPatches+p); 
        vortex_tube_indices_vertices.push_back(n-nPatches-nPatches+pn); 
        vortex_tube_indices_vertices.push_back(n-nPatches-nPatches+p); 
        vortex_tube_indices_vertices.push_back(n-nPatches+p); 
        vortex_tube_indices_vertices.push_back(n-nPatches+pn); 
        vortex_tube_indices_vertices.push_back(n-nPatches-nPatches+pn); 
      }
    }
  }
}
