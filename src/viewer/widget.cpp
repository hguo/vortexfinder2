#include <QMouseEvent>
#include <QMatrix4x4>
#include <QDebug>
#include <OpenGL/glu.h>
#include "widget.h"
#include "common/DataInfo.pb.h"

using namespace std; 

CGLWidget::CGLWidget(const QGLFormat& fmt, QWidget *parent, QGLWidget *sharedWidget)
  : QGLWidget(fmt, parent, sharedWidget), 
    _fovy(30.f), _znear(0.1f), _zfar(10.f), 
    _eye(0, 0, 2.5), _center(0, 0, 0), _up(0, 1, 0)
{
}

CGLWidget::~CGLWidget()
{
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

void CGLWidget::paintGL()
{
  glClearColor(1, 1, 1, 0); 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

  QMatrix4x4 projmatrix, mvmatrix; 
  projmatrix.setToIdentity(); 
  projmatrix.perspective(_fovy, (float)width()/height(), _znear, _zfar); 
  mvmatrix.setToIdentity();
  mvmatrix.lookAt(_eye, _center, _up);
  mvmatrix.rotate(_trackball.getRotation());
  mvmatrix.scale(_trackball.getScale());
  mvmatrix.scale(0.02);
#if 0
  mvmatrix.scale(1.0/_ds->dims()[0]); 
  mvmatrix.translate(-0.5*_ds->dims()[0], -0.5*_ds->dims()[1], -0.5*_ds->dims()[2]);
#endif

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glLoadMatrixd(projmatrix.data()); 
  glMatrixMode(GL_MODELVIEW); 
  glLoadIdentity(); 
  glLoadMatrixd(mvmatrix.data()); 

#if 1
  // render tube
  glEnable(GL_DEPTH_TEST); 
  glEnable(GL_LIGHTING); 
  glEnable(GL_LIGHT0); 

  glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT); 
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_NORMAL_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 

  glVertexPointer(3, GL_FLOAT, 0, tube_vertices.data()); 
  glNormalPointer(GL_FLOAT, 0, tube_normals.data()); 
  glColorPointer(3, GL_FLOAT, 0, tube_colors.data()); 
  glDrawElements(GL_TRIANGLES, tube_indices_vertices.size(), GL_UNSIGNED_INT, tube_indices_vertices.data()); 

  glPopClientAttrib(); 
  glDisable(GL_LIGHTING);
  // end render tubes
#else 
  glEnableClientState(GL_VERTEX_ARRAY); 
  glEnableClientState(GL_COLOR_ARRAY); 
  glVertexPointer(3, GL_FLOAT, 0, line_vertices.data()); 
  glColorPointer(4, GL_FLOAT, 4*sizeof(GLfloat), line_colors.data());
  
  glMultiDrawArrays(GL_LINE_STRIP, line_indices.data(), line_vert_count.data(), line_vert_count.size());
  // glMultiDrawArrays(GL_POINTS, line_indices.data(), line_vert_count.data(), line_vert_count.size());

  glDisableClientState(GL_COLOR_ARRAY); 
  glDisableClientState(GL_VERTEX_ARRAY); 
#endif

  CHECK_GLERROR(); 
}

void CGLWidget::LoadVortexObjects(const std::string& filename)
{
  ReadVortexOjbects(filename, _vortex_objects); 

  for (int j=0; j<_vortex_objects.size(); j++) { // iterate over v_objs
  float c[4] = {1, 0, 0, 1}; // color;
#if 1
    switch (j%7) {
    case 0: c[0]=0; c[1]=0; c[2]=0; c[3]=1; break;
    case 1: c[0]=0; c[1]=0; c[2]=1; c[3]=1; break;
    case 2: c[0]=0; c[1]=1; c[2]=0; c[3]=1; break;
    case 3: c[0]=0; c[1]=1; c[2]=1; c[3]=1; break;
    case 4: c[0]=1; c[1]=0; c[2]=0; c[3]=1; break;
    case 5: c[0]=1; c[1]=0; c[2]=1; c[3]=1; break;
    case 6: c[0]=1; c[1]=1; c[2]=0; c[3]=1; break;
    default: break; 
   }
#endif
    for (int k=0; k<_vortex_objects[j].size(); k++) { //iterator over lines
      int vertCount = _vortex_objects[j][k].size()/3;  
      
      for (std::vector<double>::iterator it = _vortex_objects[j][k].begin(); 
          it != _vortex_objects[j][k].end(); it ++) {
        line_vertices.push_back(*it); 
      }
      
      for (int l=0; l<vertCount; l++)  
        for (int m=0; m<4; m++) 
          line_colors.push_back(c[m]); 

      line_vert_count.push_back(vertCount); 
      // fprintf(stderr, "vert_count=%d\n", vertCount);
    }
  }
  
  int cnt = 0; 
  for (int i=0; i<line_vert_count.size(); i++) {
    line_indices.push_back(cnt); 
    cnt += line_vert_count[i]; 
  }

  updateTubes(20, 0.5); 
}

////////////////
void CGLWidget::updateTubes(int nPatches, float radius) 
{
  tube_vertices.clear(); 
  tube_normals.clear(); 
  tube_colors.clear(); 
  tube_indices_lines.clear(); 
  tube_indices_vertices.clear(); 

  for (int i=0; i<line_vert_count.size(); i++) {
    if (line_vert_count[i] < 2) continue; 
     
    int first = line_indices[i]; 
    QVector3D N0; 
    for (int j=1; j<line_vert_count[i]-1; j++) {
      QVector3D P0 = QVector3D(line_vertices[(first+j-1)*3], line_vertices[(first+j-1)*3+1], line_vertices[(first+j-1)*3+2]); 
      QVector3D P  = QVector3D(line_vertices[(first+j)*3], line_vertices[(first+j)*3+1], line_vertices[(first+j)*3+2]); 
      QVector3D color = QVector3D(line_colors[(first+j)*4], line_colors[(first+j)*4+1], line_colors[(first+j)*4+2]); 

      QVector3D T = (P - P0).normalized(); 
      QVector3D N = QVector3D(-T.y(), T.x(), 0.0).normalized(); 
      QVector3D B = QVector3D::crossProduct(N, T); 

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

          tube_vertices.push_back(coord.x()); 
          tube_vertices.push_back(coord.y()); 
          tube_vertices.push_back(coord.z()); 
          // tube_vertices.push_back(1); 
          tube_normals.push_back(normal.x()); 
          tube_normals.push_back(normal.y()); 
          tube_normals.push_back(normal.z()); 
          tube_colors.push_back(color.x()); 
          tube_colors.push_back(color.y()); 
          tube_colors.push_back(color.z()); 
          tube_indices_lines.push_back(j); 
        }
      }

      for (int p=0; p<nPatches; p++) {
        int n = tube_vertices.size()/3; 
        int pn = (p+1)%nPatches; 
        tube_indices_vertices.push_back(n-nPatches+p); 
        tube_indices_vertices.push_back(n-nPatches-nPatches+pn); 
        tube_indices_vertices.push_back(n-nPatches-nPatches+p); 
        tube_indices_vertices.push_back(n-nPatches+p); 
        tube_indices_vertices.push_back(n-nPatches+pn); 
        tube_indices_vertices.push_back(n-nPatches-nPatches+pn); 
      }
    }
  }
}
