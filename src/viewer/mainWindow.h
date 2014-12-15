#ifndef _MAINWINDOW_H
#define _MAINWINDOW_H

#include "ui_mainWindow.h"

class CMainWindow : public QMainWindow, public Ui::MainWindow
{
  Q_OBJECT

public:
  CMainWindow(QWidget* = NULL);
  ~CMainWindow();

protected:
};

#endif
