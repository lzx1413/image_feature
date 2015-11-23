#ifndef ENCODE2BINARY_HPP
#define ENCODE2BINARY_HPP
#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"
using namespace cv;
/**@brief this function use cuda::gemm function, so it will take some time to upload data if you 
*just want to encode some rawfeaures,you can upload the model as GpuMat at first.
*@param rawfeature the data to be encoded(),which should have CV_32FC1 , CV_64FC1 , CV_32FC2 , or
*CV_64FC2 type.
*@param model the encode model matrix the type is same with the rawfeature
*@param result the binary code after encoding 
*/
void encode2Binary(InputArray &rawfeature, InputArray& model, Mat &result)
{
	cv::cuda::gemm(rawfeature, model, 1.0, Mat(), 0.0, result);
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			if (result.at<float>(i, j) >= 0)
			{
				result.at<float>(i, j) = 1;
			}
			else
			{
				result.at<float>(i, j) = 0;
			}
		}
	}
}

#endif // ENCODE2BINARY_HPP

