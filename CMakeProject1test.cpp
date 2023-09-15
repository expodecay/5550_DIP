// CMakeProject1test.cpp : Defines the entry point for the application.
//

#include "CMakeProject1test.h"

#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


using namespace std;

int main()
{
	cv::Mat full_img, new_image;
	string file_location = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/a-The-original-Lena-image-b-The-Lena-face-Test-Image.ppm";
	full_img = cv::imread(cv::samples::findFile(file_location));

	const int n_Channel = 1;
	const int n_Dimensions = 2;
	//int mySizes[n_Dimensions] = {512, 512};
	int mySizes[n_Dimensions] = { full_img.cols, full_img.rows };

	new_image = cv::Mat::zeros(n_Dimensions, mySizes, uchar(n_Channel));


	uint8_t* myData = full_img.data;
	int width = full_img.cols;
	int height = full_img.rows;
	int _stride = full_img.step;//in case cols != strides

	std::cout << "size " << full_img.size() << " width " << width << " height " << height << " step " << _stride << endl;
	cout << "full_img value at 00: " << (int)full_img.at<uchar>(0, 1) << endl;
	//cout << full_img << endl;

	//cv::Mat new_image(int width, int height, int uint8_t, void* myData, size_t _stride);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t val = myData[i * _stride + j];
			new_image.at<uint8_t>(j, i) = 255;
			//cout << typeid(full_img).name() << endl;
		}
	}


	//cv::resize()
	cv::imshow("Full Image", full_img);
	cv::imshow("New Image", new_image);
	cv::waitKey(0);
	return 0;
}
