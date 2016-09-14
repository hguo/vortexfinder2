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
#include "common/VortexTransition.h"

#ifdef WITH_ROCKSDB
#include <rocksdb/db.h>
#endif

namespace ILines {class ILRender;}

class QMConnector; 
class QMouseEvent;
class QKeyEvent; 
class QWheelEvent; 

class PBDataInfo;

class GLGPUDataset;
struct ctx_rc;

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

  void LoadVortexLines();
  void LoadVortexLinesFromTextFile(const std::string& filename); // legacy
  void LoadVortexLines2D(); // special for 2D simulation
  void LoadFieldLines(const std::string& filename);
  void LoadInclusionsFromTextFile(const std::string& filename);

  void SetData(const std::string& dataname, int ts, int tl);
#if WITH_ROCKSDB
  void SetDB(rocksdb::DB* db);
#endif 
  void LoadTimeStep(int t);

  void SetVortexTransition(const VortexTransition* vt);

  void Clear();
  
  void OpenGLGPUDataset();

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL();
  void initIL();

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

  void renderMDS();
  void renderVortexIds();
  void renderVortexLines(); 
  void renderVortexPoints(); 
  void renderVortexTubes();
  void renderVortexArrows();
  void renderInclusions();
  void renderIsosurfaces();
  
  void updateVortexTubes(int nPatches, float radius); 

  void renderFieldLines();

protected: 
  void extractIsosurfaces();

private:
  CGLTrackball _trackball;
  QMatrix4x4 _projmatrix, _mvmatrix; 

private: //data
  // std::vector<VortexLine> _vortex_liness;
  // std::vector<FieldLine> _fieldlines;
  // PBDataInfo _data_info;
  std::string _dataname;
  int _timestep;
  int _ts, _tl;

  const VortexTransition *_vt;

private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

  int _vortex_render_mode;
  bool _toggle_inclusions;
  bool _toggle_history;
  bool _toggle_il;
  bool _toggle_ids;
  bool _toggle_bezier;
  bool _toggle_vip;
  bool _toggle_video;

  QSet<int> _vips;

private: // IL render
  ILines::ILRender *_ilrender;

private: // vortex line rendering
  std::vector<GLfloat> v_line_vertices;
  std::vector<GLubyte> v_line_colors; 
  std::vector<GLsizei> v_line_vert_count; 
  std::vector<GLint> v_line_indices; 
  
  std::vector<GLfloat> vortex_tube_vertices, vortex_tube_normals;
  std::vector<GLubyte> vortex_tube_colors; 
  std::vector<GLuint> vortex_tube_indices_lines, vortex_tube_indices_vertices;

private: //HDR
  typedef struct {
    int frame;
    float B[3];
    float Kx; // Kx
    float Jxext;
    float V; // voltage
    } vfgpu_hdr_t;
  std::vector<vfgpu_hdr_t> vfgpu_hdrs;

private: // MDS
  std::vector<float> v_mds_coords;

private: // history line render
  int h_max;
  QVector<std::vector<GLfloat> > h_line_vertices;
  QVector<std::vector<GLubyte> > h_line_colors;
  QVector<std::vector<GLsizei> > h_line_vert_count;
  QVector<std::vector<GLint> > h_line_indices;

  void addCurrentLineToHistory();
  void correctHistoryAlpha();
  void clearHistory();
  void renderHistoryVortexLines();

private: // isosurface rendering
  std::vector<GLfloat> s_triangle_vertices, s_triangle_normals;
  std::vector<GLuint> s_triangle_indices;
  std::vector<GLfloat> s_triangle_vertices1, s_triangle_normals1;
  std::vector<GLuint> s_triangle_indices1;

private: // volume rendering
  struct ctx_rc *_rc;
  float *_rc_fb;

private: // arros (cones)
  QVector<QVector3D> _cones_pos, _cones_dir;
  QVector<QColor> _cones_color;

private: // fieldline rendering
  std::vector<GLfloat> f_line_vertices;
  std::vector<GLubyte> f_line_colors; 
  std::vector<GLsizei> f_line_vert_count; 
  std::vector<GLint> f_line_indices; 

private: // id rendering
  QVector<int> _vids;
  QVector<QVector3D> _vids_coord;
  QVector<QColor> _vids_colors;
  QVector<float> _vids_speed;

private: // GLGPU
  GLGPUDataset *_ds;
#if WITH_ROCKSDB
  rocksdb::DB *_db;
#endif
}; 

#endif
