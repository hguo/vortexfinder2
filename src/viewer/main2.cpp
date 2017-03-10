#include <QApplication>
#include <QFileDialog>
// #include <GLUT/glut.h>
#include "widget.h"
#include "mainWindow.h"

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <dbname>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // DB
  const std::string dbname = argv[1];
  rocksdb::DB* db;
  rocksdb::Options options;
  // rocksdb::Status status = rocksdb::DB::Open(options, argv[1], &db);
  rocksdb::Status status = rocksdb::DB::OpenForReadOnly(options, argv[1], &db);

  // VT
  VortexTransition vt;
  vt.LoadFromDB(db);
  // vt.ConstructSequence();
  vt.PrintSequence();

  // QT
  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  CGLWidget *widget = new CGLWidget;
  widget->show();
  widget->SetDB(db);
  widget->SetData(dbname, 0, vt.NTimesteps());
  // widget->SetData(dataname, ts, tl);
  // widget->OpenGLGPUDataset();
  // widget->SetVortexTransition(&vt);
  widget->LoadTimeStep(0);

  return app.exec(); 
}

