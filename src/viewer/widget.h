#ifndef _WIDGET_H
#define _WIDGET_H

#include <QGLWidget>
#include <QList>
#include <cmath>
#include "trackball.h"
#include "extractor/vortex_object.h"

class QMConnector; 
class QMouseEvent;
class QKeyEvent; 
class QWheelEvent; 

/* 
 * \class   CGLWidget
 * \author  Hanqi Guo
 * \brief   A light-weight Qt-based vortex viewer
*/
class CGLWidget : public QGLWidget
{
  Q_OBJECT

public:
  CGLWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL); 
  ~CGLWidget(); 

  void LoadVortexObjects(/* const std::string filename */); // TODO

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL(); 

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

  void renderCoreLines(); 
  void renderCoreTubes(); 
  void updateTubes(int nPatches, float radius); 

private:
  CGLTrackball _trackball;

private: //data
  std::vector<VortexObject<> > _vortex_objects;
 
private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

private: // tube rendering
  std::vector<GLfloat> line_vertices, line_colors; 
  std::vector<GLsizei> line_vert_count; 
  std::vector<GLint> line_indices; 
  
  std::vector<GLfloat> tube_vertices, tube_normals, tube_colors; 
  std::vector<GLuint> tube_indices_lines, tube_indices_vertices;
}; 

#endif
