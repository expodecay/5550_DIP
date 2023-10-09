#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QPixmap>

#include <iostream>
#include "DIP_tools.h"


MainWindow::MainWindow(QWidget* parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    QPixmap pix2("C:/Users/rickr/Documents/Repos/5550_DIP/images/lenaTest.png");
    ui->label_pic2->setPixmap(pix2);

    QPushButton* button = new QPushButton("&Download", this);

    /*connect(ui->pushButton_2, SIGNAL(clicked()), SLOT(showPhoto()));*/
    connect(ui->Nearest_Neighbor_Interpolation, SIGNAL(clicked()), SLOT(NearestNeighborInterpolationQT()));
    connect(ui->Histogram_Equalization, SIGNAL(clicked()), SLOT(GlobalHistogramEqualizationQT()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showPhoto()
{
    QPixmap pix("C:/Users/rickr/Documents/Repos/5550_DIP/images/lena.png");
    ui->label_pic->setPixmap(pix);
}

void MainWindow::goodby()
{
    std::cout << "by" << std::endl;
}

void MainWindow::NearestNeighborInterpolationQT()
{
    NearestNeighborInterpolation();
    QPixmap pix3("C:/Users/rickr/Documents/Repos/5550_DIP/output/nearest_neighbor_image.png");
    ui->label_pic->setPixmap(pix3);
}
void MainWindow::GlobalHistogramEqualizationQT()
{
    GlobalHistogramEqualization();
    QPixmap pix3("C:/Users/rickr/Documents/Repos/5550_DIP/output/histogram_equalization_image.png");
    ui->label_pic->setPixmap(pix3);
}