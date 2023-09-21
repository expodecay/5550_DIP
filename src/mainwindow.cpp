#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QPixmap>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPixmap pix("C:/Users/rickr/Documents/Repos/5550_DIP/lena.png");
    ui->label_pic->setPixmap(pix); 
}

MainWindow::~MainWindow()
{
    delete ui;
}
