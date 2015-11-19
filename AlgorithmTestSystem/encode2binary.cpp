#include "encode2binary.h"
/**
*this function is going to encode the data from float to binary
*@param rawfeature the input data ,a line respresent a single data
*@param modle the projection matrix
*@param result the output binary data a line respresent a single data
*/
void encode2Binary( vector<float>rawfeature, Mat& model,Mat &result)
{
	Mat raw_feature = Mat(1,rawfeature.size(), CV_32FC1, rawfeature.data());
	do_metric(model,raw_feature,result);
	for (int i = 0; i < result.cols;++i)
	{
		if (result.at<float>(0, i) >= 0)
		{
			result.at<float>(0, i) = 1;
		}
		else
			result.at<float>(0, i) = 0;
	}
}

void encode2Binary(Mat &rawfeature, Mat& model, Mat &result)
{
	do_metric(model, rawfeature, result);
	for (int i = 0; i < result.rows;++i)
	{
		for (int j = 0; j < result.cols;++j)
		{
			if (result.at<float>(i,j)>=0)
			{
				result.at<float>(i,j) = 1;
			}
			else
			{
				result.at<float>(i, j) = 0;
			}
		}
	}

}


