#include <QApplication>
#include "storyLineWidget.h"

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <filename> <ts> <tl>\n", argv[0]);
    return EXIT_FAILURE;
  }
  
  const std::string filename = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]), 
            span = 1;

  QApplication app(argc, argv); 

  QGLFormat fmt = QGLFormat::defaultFormat();
  fmt.setSampleBuffers(true);
  fmt.setSamples(16); 
  QGLFormat::setDefaultFormat(fmt); 

  CStorylineWidget *widget = new CStorylineWidget;
  widget->show();

  return app.exec(); 
}
