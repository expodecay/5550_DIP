// CMakeProject1test.cpp : Defines the entry point for the application.
//
#include <QApplication>
#include "DIP_tools.h"

#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <CLI/CLI.hpp>
#include <mainwindow.h>

using namespace std;
static void insert()
{
	std::cout << "hi" << std::endl;
}

int main(int argc, char* argv[])
{
	
	QApplication app(argc, argv);
	
		MainWindow* window = new MainWindow();
		window->show();
		insert();
		return app.exec();
	// 
	//CLI::App app{"CMakeProject1test"};

	//string image_path = "C:/Users/rickr/Documents/Repos/5550_DIP/images/lena.png";

	//app.add_option("-i,--image", image_path, "location of input image");
	//CLI11_PARSE(app, argc, argv) // commandline extraction

	//cv::Mat full_img, bit_shifted_image, one_sixteenth_image, upscale_image, nearest_neighbor_image, linear_image, bilinear_image;
	////string file_location = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/lena.png";
	//full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);

	//const int n_Channel = 1;
	//const int n_Dimensions = 2;
	////int mySizes[n_Dimensions] = {512, 512};
	//int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };

	//bit_shifted_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	//nearest_neighbor_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	//linear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	//bilinear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);


	//uint8_t* myData = full_img.data;// single dimensional array
	//int width = full_img.cols;
	//int height = full_img.rows;
	//int _stride = full_img.step;//in case cols != strides

	//std::cout << "size " << full_img.size() << " width " << width << " height " << height << " step " << _stride << endl;
	//cout << "full_img value at 00: " << (int)full_img.at<uchar>(0, 1) << endl;

	//// Image copy
	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		uint8_t val = myData[i * _stride + j];

	//		bit_shifted_image.at<uint8_t>(i,j) = full_img.at<uint8_t>(i,j);

	//	}
	//}

	//// cv mat with one-sixteenth it's full size
	//int halfSize[n_Dimensions] = { full_img.rows/16, full_img.cols/16 };
	//one_sixteenth_image = cv::Mat::zeros(n_Dimensions, halfSize, CV_8U);

	//// decimation - taking every other line
	//for (int i = 0; i < height; i+=16)
	//{
	//	for (int j = 0; j < width; j+=16) 
	//	{
	//		one_sixteenth_image.at<uint8_t>(i/16, j/16) = full_img.at<uint8_t>(i, j);
	//	}
	//}

	//// up-scale back to full size
	//upscale_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	//for (int i = 0; i < height; i += 16)
	//{
	//	for (int j = 0; j < width; j += 16)
	//	{
	//		upscale_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i/16, j/16);
	//		linear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);
	//		bilinear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);
	//	}
	//}

	//// nearest neighbor upscale copy
	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		uint8_t val = myData[i * _stride + j];

	//		nearest_neighbor_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);

	//	}
	//}

	//// Linear upscale
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width - 16; j += 16) {
	//		int x1 = upscale_image.at<uint8_t>(i, j);
	//		int x2 = upscale_image.at<uint8_t>(i, j + 16);
	//		int slope = (x2 - x1) / 16;
	//		int Yint = x1;

	//		for (int k = 1; k < 16; k++) {

	//			linear_image.at<uint8_t>(i, j + k) = slope * k + Yint;

	//			//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;


	//			//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
	//		}
	//	}
	//}

	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width - 16; j += 16) {

	//		for (int k = 1; k < 16; k++) {
	//			//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
	//			//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;

	//			linear_image.at<uint8_t>(j + k, i) = linear_image.at<uint8_t>(j + k - 1, i);
	//		}
	//	}
	//}
	//
	//// Bilinear upscale
	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width - 16; j += 16) {
	//		int x1 = upscale_image.at<uint8_t>(i, j);
	//		int x2 = upscale_image.at<uint8_t>(i, j + 16);
	//		int slope = (x2 - x1)/ 16;
	//		int Yint = x1;

	//		for (int k = 1; k < 16; k++) {

	//			bilinear_image.at<uint8_t>(i, j + k) = slope * k + Yint;

	//			//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;


	//			//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
	//		}
	//	}
	//}

	//for (int i = 0; i < height; i++) {
	//	for (int j = 0; j < width - 16; j += 16) {
	//		int y1 = bilinear_image.at<uint8_t>(j, i);
	//		int y2 = bilinear_image.at<uint8_t>(j + 16, i);
	//		int slope = (y2 - y1) / 16;
	//		int Yint = y1;

	//		for (int k = 1; k < 16; k++) {
	//			//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
	//			//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;

	//			bilinear_image.at<uint8_t>(j + k, i) = slope * k + Yint;
	//		}
	//	}
	//}
	//

	//// 1bit

	//	/*for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{

	//			if (bit_shifted_image.at<uint8_t>(i, j) < 128) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 0; 
	//			}
	//			if (bit_shifted_image.at<uint8_t>(i, j) > 128) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 255;
	//			}

	//		}
	//	}*/

	//	//2bit
	//	/*for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{

	//			if (bit_shifted_image.at<uint8_t>(i, j) < 64) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 0;
	//			}
	//			if (bit_shifted_image.at<uint8_t>(i, j) > 64 && bit_shifted_image.at<uint8_t>(i, j) < 128) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 64;
	//			}
	//			if (bit_shifted_image.at<uint8_t>(i, j) > 128 && bit_shifted_image.at<uint8_t>(i, j) < 192) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 128;
	//			}
	//			if (bit_shifted_image.at<uint8_t>(i, j) > 192 && bit_shifted_image.at<uint8_t>(i, j) < 255) {
	//				bit_shifted_image.at<uint8_t>(i, j) = 192;
	//			}
	//		}
	//	}*/

	//
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{
	//			int n_bits = 2;
	//			int max_bin = pow(2, n_bits);
	//			int interval_size = 256 / max_bin;

	//			for (int k = 0; k <= max_bin; k++) {
	//				if (bit_shifted_image.at<uint8_t>(i, j) < interval_size) {
	//					bit_shifted_image.at<uint8_t>(i, j) = 0;
	//				}
	//				if (bit_shifted_image.at<uint8_t>(i, j) > interval_size*k && bit_shifted_image.at<uint8_t>(i, j) < (interval_size*k + interval_size) ) {
	//					bit_shifted_image.at<uint8_t>(i, j) = interval_size*k ;
	//				}
	//			}
	//		}
	//	}
	//
	//

	//	/*for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{

	//			bit_shifted_image.at<uint8_t>(i, j) = (uint8_t)floor(double(bit_shifted_image.at<uint8_t>(i, j)) * 0.5);

	//		}
	//	}*/
	//


	//cv::imshow("Full Image", full_img);
	//cv::imshow("Bit shifted Image", bit_shifted_image);
	//cv::imshow("Half Image", one_sixteenth_image);
	//cv::imshow("nearest_neighbor Image", nearest_neighbor_image);
	//cv::imshow("Linear Image", linear_image);
	//cv::imshow("Bilinear Image", bilinear_image);
	//cv::waitKey(0);
	//return 0;
}
