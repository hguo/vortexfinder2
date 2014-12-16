#ifndef _WIDGET_H
#define _WIDGET_H

#include <QGLWidget>
#include <QList>
#include <cmath>
#include "def.h"
#include "trackball.h"

class QMConnector; 
class QMouseEvent;
class QKeyEvent; 
class QWheelEvent; 

class PBDataInfo;

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

  void LoadVortexObjects(const std::string& filename); 
  void LoadFieldLines(const std::string& filename);

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL(); 

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

  void renderVortexLines(); 
  void renderVortexTubes(); 
  void updateVortexTubes(int nPatches, float radius); 

  void renderFieldLines();

private:
  CGLTrackball _trackball;

private: //data
  // std::vector<VortexObject> _vortex_objects;
  // std::vector<FieldLine> _fieldlines;
  PBDataInfo *_data_info;

private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

private: // vortex line rendering
  std::vector<GLfloat> v_line_vertices, v_line_colors; 
  std::vector<GLsizei> v_line_vert_count; 
  std::vector<GLint> v_line_indices; 
  
  std::vector<GLfloat> vortex_tube_vertices, vortex_tube_normals, vortex_tube_colors; 
  std::vector<GLuint> vortex_tube_indices_lines, vortex_tube_indices_vertices;

private: // fieldline rendering
  std::vector<GLfloat> f_line_vertices, f_line_colors; 
  std::vector<GLsizei> f_line_vert_count; 
  std::vector<GLint> f_line_indices; 
}; 

#endif
