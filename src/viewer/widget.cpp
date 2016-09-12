#include "ilines/ILRender.h"
#include <QMouseEvent>
#include <QFileDialog>
#include <QInputDialog>
#include <QDebug>
#include <fstream>
#include <iostream>
#include "widget.h"
#include "common/VortexLine.h"
#include "common/FieldLine.h"
#include "common/Utils.hpp"
#include "io/GLGPU3DDataset.h"

#ifdef WITH_VTK
#include <vtkNew.h>
#include <vtkImageImport.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyDataMapper.h>
#include <vtkCell.h>
#include <vtkTriangle.h>
#include <vtkPointData.h>
#include <vtkIdList.h>
#include <vtkSmartPointer.h>
#endif

#ifdef WITH_CUDA
#undef WITH_CUDA
// #include "volren/rc.h"
#endif

#ifdef __APPLE__
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glu.h>
#include <GL/glut.h>
#endif

CGLWidget::CGLWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget)
  : QGLWidget(fmt, parent, sharedWidget), 
    _fovy(30.f), _znear(0.1f), _zfar(10.f), 
    _eye(0, 0, 2.5), _center(0, 0, 0), _up(0, 1, 0), 
    _vortex_render_mode(0), 
    _toggle_inclusions(false),
    _toggle_history(false),
    _toggle_il(false),
    _toggle_ids(true),
    _toggle_bezier(false),
    _toggle_vip(false),
    _toggle_video(false),
    _ts(0), _tl(0), 
    _rc(NULL), _rc_fb(NULL),
    _ds(NULL), _vt(NULL),
    h_max(0)
{
  _ilrender = new ILines::ILRender;

  // _vips << 39 << 40 << 43 << 44;
  // _vips << 3 << 15 << 16 << 17 << 18 << 19; 
  _vips << 2;
}

CGLWidget::~CGLWidget()
{
  delete _ilrender;
  if (_ds != NULL)
    delete _ds;

#ifdef WITH_CUDA
  rc_destroy_ctx(&_rc);
  free(_rc_fb);
#endif
}

void CGLWidget::SetData(const std::string& dataname, int ts, int tl)
{
  _dataname = dataname;
  _ts = ts; 
  _tl = tl;
}

void CGLWidget::SetVortexTransition(const VortexTransition *vt)
{
  _vt = vt;
}

void CGLWidget::OpenGLGPUDataset()
{
  _ds = new GLGPU3DDataset;
  _ds->OpenDataFile(_dataname);
}

void CGLWidget::LoadTimeStep(int t)
{
  if (t<_ts || t>=_ts+_tl) return;
  _timestep = t;
  
  Clear();
  LoadVortexLines();
  
  if (_ds != NULL) {
    _ds->LoadTimeStep(t, 0);
    extractIsosurfaces();
  }
}

void CGLWidget::LoadVortexLines2D()
{
  QMap<int, QVector<float> > lines;
  QMap<int, QColor> colors;
  const float delta = 0.1;

  for (int t=_ts; t<_ts+_tl; t++) {
    std::stringstream ss;
    ss << _dataname << ".vlines." << t;
    const std::string filename = ss.str();
  
    std::string info_bytes;
    std::vector<VortexLine> vortex_liness;
    diy::unserializeFromFile(filename, vortex_liness);
    // if (!::LoadVortexLines(vortex_liness, info_bytes, filename))
    //   continue;
    
    // if (info_bytes.length()>0) 
    //   _data_info.ParseFromString(info_bytes);

    for (int i=0; i<vortex_liness.size(); i++) {
      const int gid = _vt->lvid2gvid(t, vortex_liness[i].id);
      unsigned char r, g, b;
      _vt->SequenceColor(gid, r, g, b);
      lines[gid] << *(vortex_liness[i].begin())
                 << *(vortex_liness[i].begin()+1)
                 << t*delta - (_tl*delta*0.5);
      colors[gid] = QColor(r, g, b);
    }
  }
  
  // const float O[2] = {_data_info.ox(), _data_info.oy()},
  //              L[2] = {_data_info.lx(), _data_info.ly()};
  int vertCount = 0;

  foreach (int gid, lines.keys()) {
    const QVector<float>& line = lines[gid];
    QColor c = colors[gid];
    QVector3D p0;
    for (int i=0; i<line.size()/3; i++) {
      QVector3D p(line[i*3], line[i*3+1], line[i*3+2]);
      
      v_line_vertices.push_back(p.x()); 
      v_line_vertices.push_back(p.y()); 
      v_line_vertices.push_back(p.z()); 
      v_line_colors.push_back(c.red()); 
      v_line_colors.push_back(c.green()); 
      v_line_colors.push_back(c.blue()); 
      v_line_colors.push_back(255); 

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
  }
  
  updateVortexTubes(20, 0.3); 
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
  case Qt::Key_Left:
    LoadTimeStep(_timestep - 1);
    addCurrentLineToHistory();
    updateGL();
    break;

  case Qt::Key_Period: 
  case Qt::Key_Right:
    LoadTimeStep(_timestep + 1);
    addCurrentLineToHistory();
    updateGL();
    break;

  case Qt::Key_G:
    {
      int f = QInputDialog::getInt(this, "go to timestep", "timestep");
      LoadTimeStep(f);
      updateGL();
      break;
    }
    break;

  case Qt::Key_T:
    _vortex_render_mode = 0;
    updateGL();
    break;

  case Qt::Key_L:
    if (e->modifiers() == Qt::ShiftModifier) 
      _vortex_render_mode = 3; // points
    else 
      _vortex_render_mode = 1;
    updateGL();
    break;

  case Qt::Key_S:
    _vortex_render_mode = 2; // isosurfaces
    updateGL();
    break;

  case Qt::Key_I:
    _toggle_inclusions = !_toggle_inclusions;
    updateGL();
    break;

  case Qt::Key_D:
    _toggle_ids = !_toggle_ids;
    updateGL();
    break;

  case Qt::Key_B:
    _toggle_bezier = !_toggle_bezier;
    Clear();
    LoadVortexLines();
    updateGL();
    break;

  case Qt::Key_V:
    _toggle_vip = !_toggle_vip;
    Clear();
    LoadVortexLines();
    updateGL();
    break;

  case Qt::Key_H:
    if (e->modifiers() == Qt::ShiftModifier) 
      clearHistory();
    else 
      _toggle_history = !_toggle_history;
    updateGL();
    break;
   
  case Qt::Key_M:
    _toggle_il = !_toggle_il;
    updateGL();
    break;

  case Qt::Key_C: // camera I/O
    if (e->modifiers() == Qt::ShiftModifier) { // save camera
      QString filename = QFileDialog::getSaveFileName(this, "save trackball", "./", "*.trac");
      if (filename.isEmpty()) break;
      _trackball.saveStatus(filename.toStdString().c_str());
    } else { // load camera
      QString filename = QFileDialog::getOpenFileName(this, "open trackball", "./", "*.trac");
      if (filename.isEmpty()) break;
      _trackball.loadStatus(filename.toStdString().c_str());
      updateGL();
    }
    break;

  case Qt::Key_P: 
  if (e->modifiers() == Qt::ShiftModifier) { // capture video
    _toggle_video = true;
    _timestep = _ts;
    LoadTimeStep(_timestep);
    updateGL();
  } else { // save to PNG
    QString filename = QFileDialog::getSaveFileName(this, "save to png", "./", "*.png");
    if (filename.isEmpty()) return;
    QImage img = grabFrameBuffer(false);
    img.save(filename);
  }

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
  glewInit();
  initIL();

  _trackball.init();

  // opengl smooth rendering
  {
    glEnable(GL_MULTISAMPLE);

    GLint bufs, samples; 
    glGetIntegerv(GL_SAMPLE_BUFFERS, &bufs); 
    glGetIntegerv(GL_SAMPLES, &samples); 

    glEnable(GL_LINE_SMOOTH); 
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); 
    
    glEnable(GL_POLYGON_SMOOTH); 
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST); 
    
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);

    glEnable(GL_BLEND); 
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }
  
  // initialze light for tubes
  {
    GLfloat ambient[]  = {0.1, 0.1, 0.1}, 
            diffuse[]  = {0.5, 0.5, 0.5}, 
            specular[] = {0.8, 0.8, 0.8}; 
    GLfloat dir[] = {0, 0, -1}; 
    GLfloat pos[] = {1, 1, 4, 1};
    GLfloat shiness = 100; 

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient); 
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse); 
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular); 
    glLightfv(GL_LIGHT0, GL_POSITION, pos); 
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir); 
    
    GLfloat light1_position[] = {-4.0, 4.0, 0.0, 1.0};
    GLfloat light1_spot_direction[] = {1.0, -1.0, 0.0};

    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
    glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, light1_spot_direction);

    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE); 
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); 

    glEnable(GL_NORMALIZE); 
    glEnable(GL_COLOR_MATERIAL); 
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular); 
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shiness); 
  }

  // initialize volume renderer
#ifdef WITH_CUDA
  {
    rc_create_ctx(&_rc);
    rc_set_kernel(_rc, RCKERNEL_FLOAT);
    rc_set_stepsize(_rc, 0.5);
    _rc_fb = (float*)malloc(sizeof(float)*2048*2048);
  }
#endif

  CHECK_GLERROR(); 
}

void CGLWidget::initIL()
{
  const float ka = 0.02f, kd = 0.8f, ks = 1.0f, gloss = 10.0f;
  //const float ka = 0.005f, kd = 0.8f, ks = 1.0f, gloss = 10.0f;
  const int texDim=256; 
  GLfloat lightDir[3] = {-1.0f, -1.0f, 6.0f}; 
  //GLfloat lightDir[3] = {-1.0f, 2.0f, 0.0f}; 

#if 0
  _ilrender->setupTextures(0.1f*ka, kd, 3.0f*ks, gloss, texDim,
                                ILines::ILLightingModel::IL_CYLINDER_PHONG, false,
                                lightDir);
#else
  _ilrender->setupTextures(0.2f*ka, kd, 3.0f*ks, 4.0f * gloss, texDim,
                                ILines::ILLightingModel::IL_CYLINDER_BLINN, false);
#endif
}

void CGLWidget::resizeGL(int w, int h)
{
  _trackball.reshape(w, h); 
  glViewport(0, 0, w, h);

#ifdef WITH_CUDA
  rc_set_viewport(_rc, 0, 0, w, h);
#endif

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

  QString s0 = QString("timestep=%1, frame=%2").arg(_timestep).arg(_vt->TimestepToFrame(_timestep));
  renderText(20, 60, s0, ft);

  // ft.setPointSize(24);
  ft.setPointSize(16);
  for (int i=0; i<_vids.size(); i++) {
    const int id = _vids[i];
    const double speed = _vids_speed[i];
    QVector3D v = _vids_coord[i];
    QString s = QString("%1").arg(id);
    if (!isnan(speed)) s = s + ": " + QString::number(speed, 'f', 2);
#if 0
    glColor3ub(_vids_colors[i].red(), _vids_colors[i].green(), _vids_colors[i].blue());
    glPushMatrix();
    glTranslatef(v.x(), v.y(), v.z());
    glutSolidSphere(0.5, 20, 20);
    glPopMatrix();
#endif

    glColor3f(0, 0, 0);
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
  glEnable(GL_LIGHT1); 
  glEnable(GL_CULL_FACE);

  glPushMatrix();
  glTranslatef(-64, -32, -8); // FIXME: hard code
  glColor4f(0.5, 0.5, 0.5, 0.4);
  for (int i=0; i<n; i++) {
    // glColor4ub(inclusions[i].c.red(), inclusions[i].c.green(), inclusions[i].c.blue(), 128);
    glPushMatrix();
    glTranslatef(inclusions[i].p.x(), inclusions[i].p.y(), inclusions[i].p.z());
    glutSolidSphere(radius, 64, 64);
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
  glEnable(GL_LIGHT1); 

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
    glTranslatef(0, 0, -1.5);
    glutSolidCone(1, 3, 12, 4); 
    glPopMatrix();
  }

  glPopAttrib();
}

void CGLWidget::renderVortexPoints()
{
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 
  glVertexPointer(3, GL_FLOAT, 0, v_line_vertices.data()); 
  glColorPointer(4, GL_UNSIGNED_BYTE, 4*sizeof(GLubyte), v_line_colors.data());

  glMultiDrawArrays(
      GL_POINTS, 
      v_line_indices.data(), 
      v_line_vert_count.data(), 
      v_line_vert_count.size());

  glDisableClientState(GL_COLOR_ARRAY); 
  glDisableClientState(GL_VERTEX_ARRAY);

  CHECK_GLERROR();
}

void CGLWidget::renderVortexLines()
{
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 
  glVertexPointer(3, GL_FLOAT, 0, v_line_vertices.data()); 
  glColorPointer(4, GL_UNSIGNED_BYTE, 4*sizeof(GLubyte), v_line_colors.data());

#if 0
  _ilrender->enableZSort(true);
  glLineWidth(3.f);
  glDepthMask(GL_FALSE);
  _ilrender->multiDrawArrays(
      v_line_indices.data(), 
      v_line_vert_count.data(), 
      v_line_vert_count.size());
  glDepthMask(GL_TRUE);
#else
  glMultiDrawArrays(
      GL_LINE_STRIP, 
      v_line_indices.data(), 
      v_line_vert_count.data(), 
      v_line_vert_count.size());
#endif

  glDisableClientState(GL_COLOR_ARRAY); 
  glDisableClientState(GL_VERTEX_ARRAY);

  CHECK_GLERROR();
}

void CGLWidget::renderHistoryVortexLines()
{
  _ilrender->enableZSort(true);
  glLineWidth(3.f);
  glEnable(GL_DEPTH_TEST);
 
  for (int i=0; i<h_line_vertices.size(); i++) {
    glEnableClientState(GL_VERTEX_ARRAY); 
    glEnableClientState(GL_COLOR_ARRAY); 
    glVertexPointer(3, GL_FLOAT, 0, h_line_vertices[i].data());
    glColorPointer(4, GL_UNSIGNED_BYTE, 4*sizeof(GLubyte), h_line_colors[i].data());

    if (_toggle_il) {
      // glDepthMask(GL_FALSE);
      _ilrender->multiDrawArrays(
          h_line_indices[i].data(),
          h_line_vert_count[i].data(), 
          h_line_vert_count[i].size());
      // glDepthMask(GL_TRUE);
    }
    else 
      glMultiDrawArrays(
          GL_LINE_STRIP, 
          h_line_indices[i].data(), 
          h_line_vert_count[i].data(), 
          h_line_vert_count[i].size());

    glDisableClientState(GL_COLOR_ARRAY); 
    glDisableClientState(GL_VERTEX_ARRAY);
  }

  CHECK_GLERROR();
}

void CGLWidget::renderVortexTubes()
{
  glPushAttrib(GL_ENABLE_BIT);

  glEnable(GL_DEPTH_TEST); 
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 
  glEnable(GL_LIGHT1); 

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

void CGLWidget::renderIsosurfaces()
{
  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_DEPTH_TEST); 
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 

  glEnable(GL_NORMALIZE);

  glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT); 
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_NORMAL_ARRAY); 

  glColor4ub(237, 28, 36, 255);
  glVertexPointer(3, GL_FLOAT, 0, s_triangle_vertices.data());
  glNormalPointer(GL_FLOAT, 0, s_triangle_normals.data());
  glDrawElements(GL_TRIANGLES, s_triangle_indices.size(), 
      GL_UNSIGNED_INT, s_triangle_indices.data()); 
  
  glColor4ub(250, 168, 25, 60);
  glVertexPointer(3, GL_FLOAT, 0, s_triangle_vertices1.data());
  glNormalPointer(GL_FLOAT, 0, s_triangle_normals1.data());
  glDrawElements(GL_TRIANGLES, s_triangle_indices1.size(), 
      GL_UNSIGNED_INT, s_triangle_indices1.data()); 

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
  glEnable(GL_DEPTH_TEST);
  glColor3f(0.f, 0.f, 0.f);
  glPushMatrix();
  // glScalef(128.f, 128.f, 64.f);
  glScalef(64.f, 64.f, 0.1f);
  glutWireCube(1.0);
  glPopMatrix();
#endif
  
  if (_toggle_history)
    renderHistoryVortexLines();

  if (_vortex_render_mode == 0)
    renderVortexTubes();
  else if (_vortex_render_mode == 1)
    renderVortexLines();
  else if (_vortex_render_mode == 2) {
    // renderVortexTubes();
    renderIsosurfaces();
  } else if (_vortex_render_mode == 3) 
    renderVortexPoints();

  if (_toggle_inclusions)
    renderInclusions();

  renderVortexArrows();
  if (_toggle_ids)
    renderVortexIds();

  renderFieldLines();

  if (_toggle_video) {
    QString filename = QString("frame-%1.png").arg(_timestep);
    qDebug() << filename;
    QImage img = grabFrameBuffer(false);
    img.save(filename);
    if (_timestep < _ts + _tl - 1)
      LoadTimeStep(_timestep + 1);
    else 
      _toggle_video = false;
    QMetaObject::invokeMethod(this, "updateGL", Qt::QueuedConnection);
  }

  CHECK_GLERROR(); 
}

void CGLWidget::LoadFieldLines(const std::string& filename)
{
#if 0
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
#endif
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
  _vids_colors.clear();
  _vids_speed.clear();

  _cones_pos.clear();
  _cones_dir.clear();
  _cones_color.clear();
}

void CGLWidget::LoadVortexLines()
{
#if WITH_ROCKSDB
  std::stringstream ss;
  ss << "v." << _vt->TimestepToFrame(_timestep);
  const std::string key = ss.str();
  std::string info_bytes, buf;

  rocksdb::Status s = _db->Get(rocksdb::ReadOptions(), key, &buf);
  std::vector<VortexLine> vortex_liness;
  diy::unserialize(buf, vortex_liness);

  fprintf(stderr, "Loaded vortex line from DB, key=%s\n", key.c_str());
#else
  std::stringstream ss;
  ss << _dataname << ".vlines." << _timestep;
  const std::string filename = ss.str();

  std::string info_bytes;
  std::vector<VortexLine> vortex_liness;
  if (!::LoadVortexLines(vortex_liness, info_bytes, filename))
    return;
  
  fprintf(stderr, "Loaded vortex line file from %s\n", filename.c_str());
#endif

  for (int i=0; i<vortex_liness.size(); i++) {
    vortex_liness[i].gid = _vt->lvid2gvid(_timestep, vortex_liness[i].id);
    _vt->SequenceColor(vortex_liness[i].gid, vortex_liness[i].r, vortex_liness[i].g, vortex_liness[i].b);
    // fprintf(stderr, "t=%d, lid=%d, gid=%d\n", _timestep, vortex_liness[i].id, vortex_liness[i].gid);
  }

  // const float O[3] = {_data_info.ox(), _data_info.oy(), _data_info.oz()},
  //              L[3] = {_data_info.lx(), _data_info.ly(), _data_info.lz()};

  int vertCount = 0; 
  for (int k=0; k<vortex_liness.size(); k++) { //iterator over lines
    if (_toggle_vip && !_vips.contains(vortex_liness[k].gid)) continue;

    if (vortex_liness[k].size()>=3) {
      _vids.push_back(vortex_liness[k].gid);
#if 0
      // search for the min(x) point
      QVector3D maxv;
      int maxi;
      for (int i=0; i<vortex_liness[k].size()/3; i++) {
        if (vortex_liness[k][i*3] < minx) {
          minx = vortex_liness[k][i*3];
          minxi = i;
        }
      }
      QVector3D pt(vortex_liness[k][minxi*3], 
                   vortex_liness[k][minxi*3+1],
                   vortex_liness[k][minxi*3+2]);
#else
      QVector3D pt(*(vortex_liness[k].begin()), 
                   *(vortex_liness[k].begin()+1),
                   *(vortex_liness[k].begin()+2));
#endif
      _vids_coord.push_back(pt);
      
      QColor color(vortex_liness[k].r, vortex_liness[k].g, vortex_liness[k].b);
      _vids_colors.push_back(color);
      _vids_speed.push_back(vortex_liness[k].moving_speed);
    }

    if (_toggle_bezier) {
      // vortex_liness[k].Flattern(O, L);
      vortex_liness[k].ToBezier();
    }

    if (vortex_liness[k].is_bezier) { // TODO: make it more graceful..
      VortexLine& vl = vortex_liness[k];
      const int span = vl.size()/4/2/2;
      // const int span = 18;

      for (int i=4*span; i<vl.size()/3; i+=4*span) {
#if 0
        QVector3D p(
            fmod1(vl[i*3]-O[0], L[0]) + O[0], 
            fmod1(vl[i*3+1]-O[1], L[1]) + O[1], 
            fmod1(vl[i*3+2]-O[2], L[2]) + O[2]);
#else
        QVector3D p(vl[i*3], vl[i*3+1], vl[i*3+2]);
#endif
        QVector3D p0(vl[i*3], vl[i*3+1], vl[i*3+2]), 
                  p1(vl[i*3+3], vl[i*3+4], vl[i*3+5]);
        QVector3D d = (p1 - p0).normalized();
        QColor color(vl.r, vl.g, vl.b);

        _cones_pos.push_back(p);
        _cones_dir.push_back(d);
        _cones_color.push_back(color);
      }

      vl.ToRegular(0.02);
      // vl.Unflattern(O, L);
    }
    
    std::vector<float>::iterator it = vortex_liness[k].begin();
    unsigned char c[3] = {vortex_liness[k].r, vortex_liness[k].g, vortex_liness[k].b};
    QVector3D p0;
    for (int i=0; i<vortex_liness[k].size()/3; i++) {
      QVector3D p(*it, *(++it), *(++it));
      it ++;
      
      v_line_vertices.push_back(p.x()); 
      v_line_vertices.push_back(p.y()); 
      v_line_vertices.push_back(p.z()); 
      v_line_colors.push_back(c[0]); 
      v_line_colors.push_back(c[1]); 
      v_line_colors.push_back(c[2]); 
      v_line_colors.push_back(255); 

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

  updateVortexTubes(20, 0.3); 
}

void CGLWidget::LoadVortexLinesFromTextFile(const std::string& filename)
{
  std::ifstream ifs; 
  ifs.open(filename.c_str());
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
    for (int m=0; m<4; m++) 
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
      GLubyte color[3] = {v_line_colors[(first+j)*4], v_line_colors[(first+j)*4+1], v_line_colors[(first+j)*4+2]}; 

      QVector3D T = (P - P0).normalized(); 
      QVector3D N = QVector3D(-T.y(), T.x(), 0.0).normalized(); 
      QVector3D B = QVector3D::crossProduct(N, T); 

      if (N.length() == 0 || isnan(N.length())) N=QVector3D(1,0,0);
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

void CGLWidget::extractIsosurfaces()
{
// #ifdef WITH_VTK
#if 0
  const float isovalue = 0.2, 
               isovalue1 = 0.6;
  const float *re = _ds->GetDataPointerRe(), 
               *im = _ds->GetDataPointerIm();
  const int count = _ds->dims()[0] * _ds->dims()[1] * _ds->dims()[2];
  float *rho = (float*)malloc(sizeof(float)*count);

  for (int i=0; i<count; i++)
    rho[i] = sqrt(re[i]*re[i] + im[i]*im[i]);

  vtkNew<vtkImageImport> import;
  import->SetDataScalarTypeToDouble();
  import->SetDataExtent(0, _ds->dims()[0]-1, 0, _ds->dims()[1]-1, 0, _ds->dims()[2]-1);
  import->SetWholeExtent(0, _ds->dims()[0]-1, 0, _ds->dims()[1]-1, 0, _ds->dims()[2]-1);
  import->SetDataOrigin(_ds->Origins()[0], _ds->Origins()[1], _ds->Origins()[2]);
  import->SetDataSpacing(_ds->CellLengths()[0], _ds->CellLengths()[1], _ds->CellLengths()[2]);
  import->SetImportVoidPointer(rho);
  import->Update();

  vtkNew<vtkMarchingCubes> surface;
  surface->SetInputData(import->GetOutput());
  surface->ComputeNormalsOn();
  surface->SetValue(0, isovalue);
  surface->Update();

  {
    vtkPolyData* poly = surface->GetOutput();

    s_triangle_vertices.clear(); 
    s_triangle_normals.clear();
    s_triangle_indices.clear();

    vtkDataArray* normals = poly->GetPointData()->GetNormals();

    for (int i=0; i<poly->GetNumberOfPoints(); i++) {
      s_triangle_vertices.push_back(poly->GetPoint(i)[0]); 
      s_triangle_vertices.push_back(poly->GetPoint(i)[1]); 
      s_triangle_vertices.push_back(poly->GetPoint(i)[2]); 
      s_triangle_normals.push_back(normals->GetComponent(i, 0));
      s_triangle_normals.push_back(normals->GetComponent(i, 1));
      s_triangle_normals.push_back(normals->GetComponent(i, 2));
    }

    for (int i=0; i<poly->GetNumberOfCells(); i++) {
      vtkSmartPointer<vtkIdList> list = vtkSmartPointer<vtkIdList>::New();
      poly->GetCellPoints(i, list);
      s_triangle_indices.push_back(list->GetId(0));
      s_triangle_indices.push_back(list->GetId(1));
      s_triangle_indices.push_back(list->GetId(2));
    }
  }
  
  surface->SetValue(0, isovalue1);
  surface->Update();
  
  {
    vtkPolyData* poly = surface->GetOutput();

    s_triangle_vertices1.clear(); 
    s_triangle_normals1.clear();
    s_triangle_indices1.clear();

    vtkDataArray* normals = poly->GetPointData()->GetNormals();

    for (int i=0; i<poly->GetNumberOfPoints(); i++) {
      s_triangle_vertices1.push_back(poly->GetPoint(i)[0]); 
      s_triangle_vertices1.push_back(poly->GetPoint(i)[1]); 
      s_triangle_vertices1.push_back(poly->GetPoint(i)[2]); 
      s_triangle_normals1.push_back(normals->GetComponent(i, 0));
      s_triangle_normals1.push_back(normals->GetComponent(i, 1));
      s_triangle_normals1.push_back(normals->GetComponent(i, 2));
    }

    for (int i=0; i<poly->GetNumberOfCells(); i++) {
      vtkSmartPointer<vtkIdList> list = vtkSmartPointer<vtkIdList>::New();
      poly->GetCellPoints(i, list);
      s_triangle_indices1.push_back(list->GetId(0));
      s_triangle_indices1.push_back(list->GetId(1));
      s_triangle_indices1.push_back(list->GetId(2));
    }
  }
  
  free(rho);
  fprintf(stderr, "isosurface extracted.\n");
#endif
}

void CGLWidget::addCurrentLineToHistory()
{
  h_line_vertices.push_back(v_line_vertices);
  h_line_colors.push_back(v_line_colors);
  h_line_vert_count.push_back(v_line_vert_count);
  h_line_indices.push_back(v_line_indices);

  if (h_line_vertices.size() >= h_max) {
    h_line_vertices.pop_front();
    h_line_colors.pop_front();
    h_line_vert_count.pop_front();
    h_line_indices.pop_front();
  }

  correctHistoryAlpha();
}

void CGLWidget::correctHistoryAlpha()
{
  const float alpha0 = 0.3, alpha1 = 0.8; 
  const int n = h_line_vertices.size();

  for (int i=0; i<n; i++) {
    const int k = h_max - n + i;
    const float a = (float)k/h_max;
    const float alpha = (1-a) * alpha0 + a * alpha1;

    for (int j=0; j<h_line_colors[i].size()/4; j++)
      h_line_colors[i][j*4+3] = alpha * 255;
  }
  // TODO
}

void CGLWidget::clearHistory()
{
  h_line_vertices.clear();
  h_line_colors.clear();
  h_line_vert_count.clear();
  h_line_indices.clear();
}
