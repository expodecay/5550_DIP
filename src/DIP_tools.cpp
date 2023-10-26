// CMakeProject1test.cpp : Defines the entry point for the application.
//e
#include "DIP_tools.h"

#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>

#include "opencv2/imgproc.hpp"


#include <CLI/CLI.hpp>


using namespace std;

//string image_path = "C:/Users/rickr/Documents/Repos/5550_DIP/images/lena.png";
string image_path = "C:/Users/rickr/Documents/Repos/5550_DIP/images/gaussian.png";
cv::Mat full_img;

const int n_Channel = 1;
const int n_Dimensions = 2;


void NearestNeighborInterpolation()
{
	std::cout << "NearestNeighborInterpolation()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat nearest_neighbor_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);

	uint8_t* myData = full_img.data;// single dimensional array
	int width = full_img.cols;
	int height = full_img.rows;
	int _stride = full_img.step;//in case cols != strides

	int halfSize[n_Dimensions] = { full_img.rows / 16, full_img.cols / 16 };
	cv::Mat one_sixteenth_image = cv::Mat::zeros(n_Dimensions, halfSize, CV_8U);

	// decimation - taking every other line
	for (int i = 0; i < height; i += 16)
	{
		for (int j = 0; j < width; j += 16)
		{
			one_sixteenth_image.at<uint8_t>(i / 16, j / 16) = full_img.at<uint8_t>(i, j);
		}
	}
	// nearest neighbor upscale copy
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			nearest_neighbor_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/nearest_neighbor_image.png", nearest_neighbor_image);
}

void GlobalHistogramEqualization()
{
	std::cout << "HistogramEqualization()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat global_histogram_equalization_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);

	int intensity[256] = { 0 };
	double probability[256] = { 0 };
	double cumulativeProbability[256] = { 0 };

	//pixelFrequency
	for (int j = 0; j < full_img.rows; j++)
		for (int i = 0; i < full_img.cols; i++)
			intensity[int(full_img.at<uchar>(j, i))]++;
	//pixelProbability
	for (int i = 0; i < 256; i++)
		probability[i] = intensity[i] / double(full_img.rows * full_img.cols);
	//cumuProbability
	cumulativeProbability[0] = probability[0];
	for (int i = 1; i < 256; i++)
		cumulativeProbability[i] = probability[i] + cumulativeProbability[i - 1];
	for (int i = 0; i < 256; i++)
		cumulativeProbability[i] = floor(cumulativeProbability[i] * 255);
	for (int j = 0; j < full_img.rows; j++)
	{
		for (int i = 0; i < full_img.cols; i++)
		{
			//int color = cumulativeProbability[int(img.at<uchar>(i, j))];
			global_histogram_equalization_image.at<uchar>(i, j) = cumulativeProbability[int(full_img.at<uchar>(i, j))];
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/global_histogram_equalization_image.png", global_histogram_equalization_image);
}

void LocalHistogramEqualization()
{
	std::cout << "LocalHistogramEqualization()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat local_histogram_equalization_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			local_histogram_equalization_image.at<uint8_t>(i,j) = full_img.at<uint8_t>(i,j);
		}
	}

	int kernelSize[n_Dimensions] = { std::clamp(9, 3, full_img.cols), std::clamp(9, 3, full_img.rows) }; //min 3x3nmax 512x512, going over & under clams to min / max

	int sub_offset_x = full_img.cols - (full_img.cols / kernelSize[0]) * kernelSize[0];
	int sub_offset_y = full_img.rows - (full_img.rows / kernelSize[1]) * kernelSize[1];

	cv::Mat kernel = cv::Mat::zeros(n_Dimensions, kernelSize, CV_8U);
	//pixelFrequency
	for (int i = kernelSize[1]; i < full_img.rows- kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols- kernelSize[0]; j++) { // for every pixel in image, hist eq over kernel
			int intensity[256] = { 0 };
			double probability[256] = { 0 };
			double cumulativeProbability[256] = { 0 };

			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2) ; k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {

					intensity[int(full_img.at<uchar>(k+i, l+j))]++;
					//pixelProbability
					for (int m = 0; m < 256; m++) {
						probability[m] = intensity[m] / double(kernel.rows * kernel.cols);
					}
					//cumuProbability
					cumulativeProbability[0] = probability[0];
					for (int n = 1; n < 256; n++) {
						cumulativeProbability[n] = probability[n] + cumulativeProbability[n - 1];
					}
					for (int p = 0; p < 256; p++) {
						cumulativeProbability[p] = round(cumulativeProbability[p] * 255);
					}
					
				}
			}
			local_histogram_equalization_image.at<uint8_t>(i, j) = cumulativeProbability[int(full_img.at<uchar>(i, j))];
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/local_histogram_equalization_image.png", local_histogram_equalization_image);

}

void SmoothingFilter()
{
	std::cout << "SmoothingFilter()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat smoothing_filter_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			smoothing_filter_image.at<uint8_t>(i, j) = full_img.at<uint8_t>(i, j);
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) }; 
	int sub_offset_x = full_img.cols - (full_img.cols / kernelSize[0]) * kernelSize[0];
	int sub_offset_y = full_img.rows - (full_img.rows / kernelSize[1]) * kernelSize[1];

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_8U);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) { 
			int average = 0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					average += full_img.at<uint8_t>(k + i, l + j) * kernel.at<uint8_t>(k + kernelSize[1] / 2, l + kernelSize[1] / 2);
				}
			}
			average /= 9;

			smoothing_filter_image.at<uint8_t>(i, j) = average;

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/smoothing_filter.png", smoothing_filter_image);
}

void MedianFilter()
{
	std::cout << "MedianFilter()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat median_filter_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			median_filter_image.at<uint8_t>(i, j) = full_img.at<uint8_t>(i, j);
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(7, 3, full_img.cols), std::clamp(7, 3, full_img.rows) };
	cv::Mat kernel = cv::Mat::zeros(n_Dimensions, kernelSize, CV_8U);

	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			std::vector<int> pixel_values;
			int median_value;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					pixel_values.push_back(full_img.at<uint8_t>(k + i, l + j));
				}
			}
			std::sort(pixel_values.begin(), pixel_values.end());
			median_value = pixel_values[pixel_values.size() / 2];
			median_filter_image.at<uint8_t>(i, j) = median_value;

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/median_filter_image.png", median_filter_image);
}

void LaplacianFilter()
{
	std::cout << "LaplacianFilter()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat laplacian_filter_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);
	//cv::Mat workspace(n_Dimensions, fullSize, CV_8U, 255);
	cv::Mat workspace = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);


	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			laplacian_filter_image.at<float>(i, j) = static_cast<float>(full_img.at<uint8_t>(i, j));
		}
	}

	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };
	cv::Mat kernel = cv::Mat::zeros(n_Dimensions, kernelSize, CV_32F);
	kernel.at<float>(0, 0) = 0;
	kernel.at<float>(0, 1) = 1;
	kernel.at<float>(0, 2) = 0;
	kernel.at<float>(1, 0) = 1;
	kernel.at<float>(1, 1) = -4;
	kernel.at<float>(1, 2) = 1;
	kernel.at<float>(2, 0) = 0;
	kernel.at<float>(2, 1) = 1;
	kernel.at<float>(2, 2) = 0;

	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			float convolved_value = 0.0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					int kernel_x_index = k + kernelSize[1] / 2;
					int kernel_y_index = l + kernelSize[1] / 2;
					float current_kernel_value = kernel.at<float>(kernel_x_index, kernel_y_index);
					float current_image_value = full_img.at<uint8_t>(kernel_x_index + i, kernel_y_index + j);
					float new_value = current_image_value * current_kernel_value;
					
					convolved_value += new_value;
				}
			}
			workspace.at<uint8_t>(i, j) = static_cast<uint8_t>(std::clamp(static_cast<float>(full_img.at<uint8_t>(i, j)) + convolved_value, 0.0f, 255.0f));
			//workspace.at<uint8_t>(i, j) = static_cast<uint8_t>(std::clamp(convolved_value, 0.0f, 255.0f));
		}
	}
	laplacian_filter_image = full_img - workspace;
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/laplacian_filter_image.png", workspace);
}

void HighBoostFilter()
{
	std::cout << "HighBoostFilter()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat high_boost_filter_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);

	cv::Mat full_copy = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);
	cv::Mat difference_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);
	cv::Mat smoothing_filter_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			smoothing_filter_image.at<float>(i, j) = static_cast<float>(full_img.at<uint8_t>(i, j));
			full_copy.at<float>(i, j) = static_cast<float>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };
	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_8U);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			float average = 0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					average += static_cast<float>(full_img.at<uint8_t>(k + i, l + j)) * static_cast<float>(kernel.at<uint8_t>(k + kernelSize[1] / 2, l + kernelSize[1] / 2));
				}
			}
			average /= 9;

			smoothing_filter_image.at<float>(i, j) = average;

		}
	}
	cv::Mat mask = cv::Mat::zeros(n_Dimensions, fullSize, CV_32F);
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			mask.at<float>(i, j) = static_cast<float>(full_img.at<uint8_t>(i, j));
		}
	}
	difference_image = mask - smoothing_filter_image;
	high_boost_filter_image = full_copy + 4.5*difference_image;
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/high_boost_filter_image.png", high_boost_filter_image);
}

void BitPlaneRemoval()
{
	std::cout << "BitPlaneRemoval()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat zero = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat one = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat two = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat three = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat four = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat five = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat six = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	cv::Mat seven = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);

	std::map<int, cv::Mat> bit_planes{ {0, zero}, {1, one}, {2, two}, {3, three}, {4, four},
		{5, five}, {6, six}, {7, seven}};

	
	for (int i=0; i < 8; ++i) {
		cv::Mat out(((full_img / (1 << i)) & 1) * 255);
		bit_planes[i] = out;
	}
	cv::Mat out(((full_img / (1 << 7)) & 1) );
	cv::Mat zero_plane = bit_planes[0];
	cv::Mat one_plane = bit_planes[1];
	cv::Mat two_plane = bit_planes[2];
	cv::Mat three_plane = bit_planes[3];
	cv::Mat four_plane = bit_planes[4];
	cv::Mat five_plane = bit_planes[5];
	cv::Mat six_plane = bit_planes[6];
	cv::Mat seven_plane = bit_planes[7];

	//cv::Mat total_image = one_plane | two_plane | three_plane | four_plane | five_plane | six_plane | seven_plane;
	//cv::Mat total_image = six_plane | seven_plane;
	cv::Mat total_image = 2 * (2 * (2 * seven_plane + six_plane) + five_plane);
	cout << "test";
}

void ArithmeticMean() {
	std::cout << "ArithmeticMean()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat arithmetic_mean_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			arithmetic_mean_image.at<uint8_t>(i, j) = full_img.at<uint8_t>(i, j);
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };
	int sub_offset_x = full_img.cols - (full_img.cols / kernelSize[0]) * kernelSize[0];
	int sub_offset_y = full_img.rows - (full_img.rows / kernelSize[1]) * kernelSize[1];

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_8U);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			int arithmean = 0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					arithmean += full_img.at<uint8_t>(k + i, l + j) * kernel.at<uint8_t>(k + kernelSize[1] / 2, l + kernelSize[1] / 2);
				}
			}
			arithmean /= kernelSize[0]*kernelSize[1];

			arithmetic_mean_image.at<uint8_t>(i, j) = arithmean;

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/arithmetic_mean_image.png", arithmetic_mean_image);
}

void GeometricMean()
{
	std::cout << "GeometricMean()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat geometric_mean_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			geometric_mean_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };
	

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double geomean = 1.0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2);
					geomean *= static_cast<double>(full_img.at<uint8_t>(k + i, l + j)) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2);
				}
			}
			geomean = pow(geomean, 0.1111);

			geometric_mean_image.at<double>(i, j) = geomean;

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/geometric_mean_image.png", geometric_mean_image);
}

void HarmonicMean()
{
	std::cout << "HarmonicMean()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat harmonic_mean_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			harmonic_mean_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };


	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double sum = 0.0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2);
					sum += 1/(static_cast<double>(full_img.at<uint8_t>(k + i, l + j)) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2));
				}
			}
			double harmean = static_cast<float>(kernelSize[0]*kernelSize[1]) / sum;

			harmonic_mean_image.at<double>(i, j) = harmean;

		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/harmonic_mean_image.png", harmonic_mean_image);
}

void ContraHarmonicMean()
{
	std::cout << "ContraHarmonicMean()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat contra_harmonic_mean_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			contra_harmonic_mean_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double numerator = 0.0;
			double denominator = 0.0;
			int Q = 0;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j);
					numerator +=  pow(static_cast<double>(full_img.at<uint8_t>(k + i, l + j)) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2), Q+1);
					denominator += pow(static_cast<double>(full_img.at<uint8_t>(k + i, l + j)) * kernel.at<double>(k + kernelSize[1] / 2, l + kernelSize[1] / 2), Q);
				}
			}
			double contraharmean = numerator / denominator;

			contra_harmonic_mean_image.at<double>(i, j) = contraharmean;
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/contra_harmonic_mean_image.png", contra_harmonic_mean_image);
}

void Max()
{
	std::cout << "Max()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat max_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			max_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double max = 0;
			std::vector<double> img_subset;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j);
					img_subset.push_back(current_value);
				}
			}
			std::sort(img_subset.begin(), img_subset.end());
			max = img_subset[8];
			max_image.at<double>(i, j) = max;
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/max_image.png", max_image);
}

void Min()
{
	std::cout << "Min()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat min_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			min_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double min = 0;
			std::vector<double> img_subset;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j);
					img_subset.push_back(current_value);
				}
			}
			std::sort(img_subset.begin(), img_subset.end());
			min = img_subset[0];
			min_image.at<double>(i, j) = min;
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/min_image.png", min_image);
}

void Midpoint()
{
	std::cout << "Midpoint()" << std::endl;
	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
	cv::Mat midpoint_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_64FC1);
	// Image copy
	for (int i = 0; i < full_img.rows; i++)
	{
		for (int j = 0; j < full_img.cols; j++)
		{
			//uint8_t val = myData[i * _stride + j];
			midpoint_image.at<double>(i, j) = static_cast<double>(full_img.at<uint8_t>(i, j));
		}
	}
	int kernelSize[n_Dimensions] = { std::clamp(3, 3, full_img.cols), std::clamp(3, 3, full_img.rows) };

	cv::Mat kernel = cv::Mat::ones(n_Dimensions, kernelSize, CV_64FC1);
	for (int i = kernelSize[1]; i < full_img.rows - kernelSize[1]; i++) {
		for (int j = kernelSize[0]; j < full_img.cols - kernelSize[0]; j++) {
			double midpoint = 0;
			std::vector<double> img_subset;
			for (int k = -(kernelSize[1] / 2); k <= (kernelSize[1] / 2); k++) {
				for (int l = -(kernelSize[1] / 2); l <= (kernelSize[1] / 2); l++) {
					double current_value = full_img.at<uint8_t>(k + i, l + j);
					img_subset.push_back(current_value);
				}
			}
			std::sort(img_subset.begin(), img_subset.end());
			midpoint = img_subset[4];
			midpoint_image.at<double>(i, j) = midpoint;
			cout << "hiiiii" << endl;
		}
	}
	cv::imwrite("C:/Users/rickr/Documents/Repos/5550_DIP/output/midpoint_image.png", midpoint_image);
}
//int main(int argc, char* argv[])
//{
//	 
//	CLI::App app{"CMakeProject1test"};
//
//	string image_path = "C:/Users/rickr/Documents/Repos/5550_DIP/images/lena.png";
//
//	app.add_option("-i,--image", image_path, "location of input image");
//	CLI11_PARSE(app, argc, argv) // commandline extraction
//
//	cv::Mat full_img, bit_shifted_image, one_sixteenth_image, upscale_image, nearest_neighbor_image, linear_image, bilinear_image;
//	//string file_location = "H:/My Drive/CPP/CS_5550_Digital_Image_Processing/Assignment_1/lena.png";
//	full_img = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_GRAYSCALE);
//
//	const int n_Channel = 1;
//	const int n_Dimensions = 2;
//	//int mySizes[n_Dimensions] = {512, 512};
//	int fullSize[n_Dimensions] = { full_img.rows, full_img.cols };
//
//	bit_shifted_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
//	nearest_neighbor_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
//	linear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
//	bilinear_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
//
//
//	uint8_t* myData = full_img.data;// single dimensional array
//	int width = full_img.cols;
//	int height = full_img.rows;
//	int _stride = full_img.step;//in case cols != strides
//
//	std::cout << "size " << full_img.size() << " width " << width << " height " << height << " step " << _stride << endl;
//	cout << "full_img value at 00: " << (int)full_img.at<uchar>(0, 1) << endl;
//
//	// Image copy
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			uint8_t val = myData[i * _stride + j];
//
//			bit_shifted_image.at<uint8_t>(i,j) = full_img.at<uint8_t>(i,j);
//
//		}
//	}
//
//	// cv mat with one-sixteenth it's full size
//	int halfSize[n_Dimensions] = { full_img.rows/16, full_img.cols/16 };
//	one_sixteenth_image = cv::Mat::zeros(n_Dimensions, halfSize, CV_8U);
//
//	// decimation - taking every other line
//	for (int i = 0; i < height; i+=16)
//	{
//		for (int j = 0; j < width; j+=16) 
//		{
//			one_sixteenth_image.at<uint8_t>(i/16, j/16) = full_img.at<uint8_t>(i, j);
//		}
//	}
//
//	// up-scale back to full size
//	upscale_image = cv::Mat::zeros(n_Dimensions, fullSize, CV_8U);
//	for (int i = 0; i < height; i += 16)
//	{
//		for (int j = 0; j < width; j += 16)
//		{
//			upscale_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i/16, j/16);
//			linear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);
//			bilinear_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);
//		}
//	}
//
//	// nearest neighbor upscale copy
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			uint8_t val = myData[i * _stride + j];
//
//			nearest_neighbor_image.at<uint8_t>(i, j) = one_sixteenth_image.at<uint8_t>(i / 16, j / 16);
//
//		}
//	}
//
//	// Linear upscale
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width - 16; j += 16) {
//			int x1 = upscale_image.at<uint8_t>(i, j);
//			int x2 = upscale_image.at<uint8_t>(i, j + 16);
//			int slope = (x2 - x1) / 16;
//			int Yint = x1;
//
//			for (int k = 1; k < 16; k++) {
//
//				linear_image.at<uint8_t>(i, j + k) = slope * k + Yint;
//
//				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
//
//
//				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
//			}
//		}
//	}
//
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width - 16; j += 16) {
//
//			for (int k = 1; k < 16; k++) {
//				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
//				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
//
//				linear_image.at<uint8_t>(j + k, i) = linear_image.at<uint8_t>(j + k - 1, i);
//			}
//		}
//	}
//	
//	// Bilinear upscale
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width - 16; j += 16) {
//			int x1 = upscale_image.at<uint8_t>(i, j);
//			int x2 = upscale_image.at<uint8_t>(i, j + 16);
//			int slope = (x2 - x1)/ 16;
//			int Yint = x1;
//
//			for (int k = 1; k < 16; k++) {
//
//				bilinear_image.at<uint8_t>(i, j + k) = slope * k + Yint;
//
//				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
//
//
//				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
//			}
//		}
//	}
//
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width - 16; j += 16) {
//			int y1 = bilinear_image.at<uint8_t>(j, i);
//			int y2 = bilinear_image.at<uint8_t>(j + 16, i);
//			int slope = (y2 - y1) / 16;
//			int Yint = y1;
//
//			for (int k = 1; k < 16; k++) {
//				//upscale_image.at<uint8_t>(i, j + k) = (upscale_image.at<uint8_t>(i, j) + upscale_image.at<uint8_t>(i, j + 16)) / 2;
//				//upscale_image.at<uint8_t>(j + k, i) = (upscale_image.at<uint8_t>(j, i) + upscale_image.at<uint8_t>(j + 16, i)) / 2;
//
//				bilinear_image.at<uint8_t>(j + k, i) = slope * k + Yint;
//			}
//		}
//	}
//	
//
//	// 1bit
//
//		/*for (int i = 0; i < height; i++)
//		{
//			for (int j = 0; j < width; j++)
//			{
//
//				if (bit_shifted_image.at<uint8_t>(i, j) < 128) {
//					bit_shifted_image.at<uint8_t>(i, j) = 0; 
//				}
//				if (bit_shifted_image.at<uint8_t>(i, j) > 128) {
//					bit_shifted_image.at<uint8_t>(i, j) = 255;
//				}
//
//			}
//		}*/
//
//		//2bit
//		/*for (int i = 0; i < height; i++)
//		{
//			for (int j = 0; j < width; j++)
//			{
//
//				if (bit_shifted_image.at<uint8_t>(i, j) < 64) {
//					bit_shifted_image.at<uint8_t>(i, j) = 0;
//				}
//				if (bit_shifted_image.at<uint8_t>(i, j) > 64 && bit_shifted_image.at<uint8_t>(i, j) < 128) {
//					bit_shifted_image.at<uint8_t>(i, j) = 64;
//				}
//				if (bit_shifted_image.at<uint8_t>(i, j) > 128 && bit_shifted_image.at<uint8_t>(i, j) < 192) {
//					bit_shifted_image.at<uint8_t>(i, j) = 128;
//				}
//				if (bit_shifted_image.at<uint8_t>(i, j) > 192 && bit_shifted_image.at<uint8_t>(i, j) < 255) {
//					bit_shifted_image.at<uint8_t>(i, j) = 192;
//				}
//			}
//		}*/
//
//	
//		for (int i = 0; i < height; i++)
//		{
//			for (int j = 0; j < width; j++)
//			{
//				int n_bits = 2;
//				int max_bin = pow(2, n_bits);
//				int interval_size = 256 / max_bin;
//
//				for (int k = 0; k <= max_bin; k++) {
//					if (bit_shifted_image.at<uint8_t>(i, j) < interval_size) {
//						bit_shifted_image.at<uint8_t>(i, j) = 0;
//					}
//					if (bit_shifted_image.at<uint8_t>(i, j) > interval_size*k && bit_shifted_image.at<uint8_t>(i, j) < (interval_size*k + interval_size) ) {
//						bit_shifted_image.at<uint8_t>(i, j) = interval_size*k ;
//					}
//				}
//			}
//		}
//	
//	
//
//		/*for (int i = 0; i < height; i++)
//		{
//			for (int j = 0; j < width; j++)
//			{
//
//				bit_shifted_image.at<uint8_t>(i, j) = (uint8_t)floor(double(bit_shifted_image.at<uint8_t>(i, j)) * 0.5);
//
//			}
//		}*/
//	
//
//
//	cv::imshow("Full Image", full_img);
//	cv::imshow("Bit shifted Image", bit_shifted_image);
//	cv::imshow("Half Image", one_sixteenth_image);
//	cv::imshow("nearest_neighbor Image", nearest_neighbor_image);
//	cv::imshow("Linear Image", linear_image);
//	cv::imshow("Bilinear Image", bilinear_image);
//	cv::waitKey(0);
//	return 0;
//}
