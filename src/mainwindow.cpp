#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QPixmap>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPixmap pix("C:/Users/rickr/Documents/Repos/5550_DIP/images/lena.png");
    ui->label_pic->setPixmap(pix); 

    QPixmap pix2("C:/Users/rickr/Documents/Repos/5550_DIP/images/lenaTest.png");
    ui->label_pic2->setPixmap(pix2);
}

MainWindow::~MainWindow()
{
    delete ui;
}
