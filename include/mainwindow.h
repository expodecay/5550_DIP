#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void showPhoto(QString path);
    void NearestNeighborInterpolationQT();
    void GlobalHistogramEqualizationQT();
    void LocalHistogramEqualizationQT();
    void SmoothingFilterQT();
    void MedianFilterQT();
    void LaplacianFilterQT();
    void HighBoostFilterQT();
    void BitPlaneRemovalQT();
    void ArithmeticMeanQT();
    void GeometricMeanQT();
    void HarmonicMeanQT();

private:
    Ui::MainWindow* ui;
};

#endif // MAINWINDOW_H
