#include <QApplication>

#include <mainwindow.h>



#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	MainWindow* window = new MainWindow();

	window->show();
	return app.exec();
}

