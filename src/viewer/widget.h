#ifndef _WIDGET_H
#define _WIDGET_H

#include <QGLWidget>
#include <QList>
#include <QVector>
#include <QVector3D>
#include <QMatrix4x4>
#include <cmath>
#include "def.h"
#include "trackball.h"
#include "common/Inclusions.h"

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

  void LoadVortexLines(const std::string& filename); 
  void LoadVortexLinesFromTextFile(const std::string& filename);
 
  void LoadFieldLines(const std::string& filename);
  
  void LoadInclusionsFromTextFile(const std::string& filename);
  
  void SetDataName(const std::string& dataname);
  void LoadTimeStep(int t);
  void Clear();

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL(); 

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

  void renderVortexIds();
  void renderVortexLines(); 
  void renderVortexTubes(); 
  void updateVortexTubes(int nPatches, float radius); 
  void renderInclusions();

  void renderFieldLines();

private:
  CGLTrackball _trackball;
  QMatrix4x4 _projmatrix, _mvmatrix; 

private: //data
  // std::vector<VortexLine> _vortex_liness;
  // std::vector<FieldLine> _fieldlines;
  PBDataInfo *_data_info;
  std::string _dataname;
  int _timestep;

private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

  int _vortex_render_mode;
  bool _enable_inclusions;

private: // vortex line rendering
  std::vector<GLfloat> v_line_vertices;
  std::vector<GLubyte> v_line_colors; 
  std::vector<GLsizei> v_line_vert_count; 
  std::vector<GLint> v_line_indices; 
  
  std::vector<GLfloat> vortex_tube_vertices, vortex_tube_normals;
  std::vector<GLubyte> vortex_tube_colors; 
  std::vector<GLuint> vortex_tube_indices_lines, vortex_tube_indices_vertices;

private: // fieldline rendering
  std::vector<GLfloat> f_line_vertices;
  std::vector<GLubyte> f_line_colors; 
  std::vector<GLsizei> f_line_vert_count; 
  std::vector<GLint> f_line_indices; 

private: // id rendering
  QVector<int> _vids;
  QVector<QVector3D> _vids_coord;
}; 

#endif
