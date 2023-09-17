// CMakeProject1test.cpp : Defines the entry point for the application.
//

#include "CMakeProject1test.h"

#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <CLI/CLI.hpp>


using namespace std;

int main(int argc, char* argv[])
{
	CLI::App app{"CMakeProject1test"};

	string image_path = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/lena.png";

	app.add_option("-i,--image", image_path, "location of input image");
	CLI11_PARSE(app, argc, argv) // commandline extraction

	cv::Mat full_img, new_image, one_sixteenth_image, upscale_image, nearest_neighbor_image, linear_image, bilinear_image;
	//string file_location = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/lena.png";
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);

	const int n_Channel = 1;
	const int n_Dimensions = 2;
	//int mySizes[n_Dimensions] = {512, 512};
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };

	new_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	nearest_neighbor_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	linear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	bilinear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);


	uint8_t* myData = full_img.data;// single dimensional array
	int width = full_img.cols;
	int height = full_img.rows;
	int _stride = full_img.step;//in case cols != strides

	std::cout << "size " << full_img.size() << " width " << width << " height " << height << " step " << _stride << endl;
	cout << "full_img value at 00: " << (int)full_img.at<uchar>(0, 1) << endl;

	// Test image copy
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t val = myData[i * _stride + j];

			new_image.at<uint8_t>(i,j) = full_img.at<uint8_t>(i,j);

		}
	}

	// cv mat with one-sixteenth it's full size
	int halfSize[n_Dimensions] = { full_img.rows/16, full_img.cols/16 };
	one_sixteenth_image = cv::Mat::zeros(n_Dimensions, halfSize, CV_8U);

	// decimation - taking every other line
	for (int i = 0; i < height; i+=16)
	{
		for (int j = 0; j < width; j+=16) 
		{
			one_sixteenth_image.at<uint8_t>(i/16, j/16) = full_img.at<uint8_t>(i, j);
		}
	}

	// up-scale back to full size
	upscale_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	for (int i = 0; i < height; i += 16)
	{
		for (int j = 0; j < width; j += 16)
		{
			upscale_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i/16, j/16);
		}
	}

	// nearest neighbor upscale copy
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t val = myData[i * _stride + j];

			nearest_neighbor_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);

		}
	}
	// linear upscale copy
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t val = myData[i * _stride + j];

			linear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);

		}
	}
	// bilinear neighbor upscale copy
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t val = myData[i * _stride + j];

			//bilinear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);

		}
	}
	
	//linear upscale
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 16; j += 16) {
			int x1 = upscale_image.at<uint8_t>(i, j);
			int x2 = upscale_image.at<uint8_t>(i, j + 16);
			int slope = (x2 - x1)/ 16;
			int Yint = x1;

			for (int k = 1; k < 16; k++) {

				upscale_image.at<uint8_t>(i, j + k) = slope * k + Yint;

				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;


				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 16; j += 16) {
			int y1 = upscale_image.at<uint8_t>(j, i);
			int y2 = upscale_image.at<uint8_t>(j + 16, i);
			int slope = (y2 - y1) / 16;
			int Yint = y1;

			for (int k = 1; k < 16; k++) {
				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;

				upscale_image.at<uint8_t>(j + k, i) = slope * k + Yint;
			}
		}
	}

	cv::imshow("Full Image", full_img);
	cv::imshow("Copy Image", new_image);
	cv::imshow("Half Image", one_sixteenth_image);
	cv::imshow("Upscale Image", upscale_image);
	cv::waitKey(0);
	return 0;
}
