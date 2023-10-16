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

    connect(ui->Nearest_Neighbor_Interpolation, SIGNAL(clicked()), SLOT(NearestNeighborInterpolationQT()));
    connect(ui->Histogram_Equalization, SIGNAL(clicked()), SLOT(GlobalHistogramEqualizationQT()));
    connect(ui->Local_Histogram, SIGNAL(clicked()), SLOT(LocalHistogramEqualizationQT()));
    connect(ui->Smoothing_Filter, SIGNAL(clicked()), SLOT(SmoothingFilterQT()));
    connect(ui->Median_Filter, SIGNAL(clicked()), SLOT(MedianFilterQT()));
    connect(ui->Laplacian_Filter, SIGNAL(clicked()), SLOT(LaplacianFilterQT()));
    connect(ui->High_Boost, SIGNAL(clicked()), SLOT(HighBoostFilterQT()));
    connect(ui->Bit_Plane_Removal, SIGNAL(clicked()), SLOT(BitPlaneRemovalQT()));
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showPhoto(QString path)
{
    QPixmap pix(path);
    ui->label_pic->setPixmap(pix);
}

void MainWindow::NearestNeighborInterpolationQT()
{
    NearestNeighborInterpolation();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/nearest_neighbor_image.png");
    
}
void MainWindow::GlobalHistogramEqualizationQT()
{
    GlobalHistogramEqualization();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/global_histogram_equalization_image.png");
}

void MainWindow::LocalHistogramEqualizationQT()
{
    LocalHistogramEqualization();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/local_histogram_equalization_image.png");
}

void MainWindow::SmoothingFilterQT()
{
    SmoothingFilter();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/smoothing_filter.png");
}

void MainWindow::MedianFilterQT()
{
    MedianFilter();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/median_filter_image.png");
}

void MainWindow::LaplacianFilterQT()
{
    LaplacianFilter();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/laplacian_filter_image.png");
}

void MainWindow::HighBoostFilterQT()
{
    HighBoostFilter();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/high_boost_filter_image.png");
}

void MainWindow::BitPlaneRemovalQT()
{
    BitPlaneRemoval();
    showPhoto("C:/Users/rickr/Documents/Repos/5550_DIP/output/bit_plane_removal.png");
}