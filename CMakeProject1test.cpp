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
	cv::Mat full_img, new_image, half_image;
	string file_location = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/lena.png";
	full_img = cv::imread(cv::samples::findFile(file_location), cv::IMREAD_GRAYSCALE);

	const int n_Channel = 1;
	const int n_Dimensions = 2;
	//int mySizes[n_Dimensions] = {512, 512};
	int mySizes[n_Dimensions] = { full_img.rows, full_img.cols };

	new_image = cv::Mat::zeros(n_Dimensions, mySizes, CV_8U);


	uint8_t* myData = full_img.data;// single dimensional array
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

			new_image.at<uint8_t>(i,j) = full_img.at<uint8_t>(i,j);

		}
	}

	// cv mat with half width and height
	int halfSize[n_Dimensions] = { full_img.rows/2, full_img.cols/2 };
	half_image = cv::Mat::zeros(n_Dimensions, halfSize, CV_8U);

	//decimation
	for (int i = 0; i < height; i+=2)
	{
		for (int j = 0; j < width; j+=2) 
		{
			half_image.at<uint8_t>(i/2, j/2) = full_img.at<uint8_t>(i, j);
		}
	}

	//cv::resize()
	cv::imshow("Full Image", full_img);
	cv::imshow("New Image", new_image);
	cv::imshow("Half Image", half_image);
	cv::waitKey(0);
	return 0;
}
