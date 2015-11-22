#include "utls.h"
#include <opencv2/cudaarithm.hpp>
#include <Eigen/Eigen>
#define USE_OPENCV_MAT 
using Matric_DDF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
void L2NormFeature(Mat& smat, int rowidx)
{
	int des_dim = smat.cols;
	//L2 norm
	float total = 0.0;
	for (int kk = 0; kk < des_dim; kk++)
	{
		total += smat.at<float>(rowidx, kk) * smat.at<float>(rowidx, kk);
	}
	total = total < 0.00001 ? 1 : sqrtf(total);
	for (int kk = 0; kk < des_dim; kk++)
		smat.at<float>(rowidx, kk) /= total;
}
void L2NormFeature(Mat& smat)
{
	for (int i = 0; i < smat.rows; i++)
		L2NormFeature(smat, i);
}
/**
*此方法用于将一条string按照给定的分隔符分成string的vector型
*@param src 输入的字符串
*@param separator 分隔符
*@param dest 分理出的string数组
*/
void split_words(const string& src, const string& separator, vector<string>& dest)
{
	dest.clear();
	string str = src;
	string substring;
	string::size_type start = 0, index;

	do
	{
		index = str.find_first_of(separator, start);
		if (index != string::npos)
		{
			substring = str.substr(start, index - start);
			dest.push_back(substring);
			start = str.find_first_not_of(separator, index);
			if (start == string::npos) return;
		}
	} while (index != string::npos);

	//the last token
	substring = str.substr(start);
	dest.push_back(substring);
}
//TODO：do understand after
void RootNormFeature(vector<float>& sdes)
{
	int des_dim = sdes.size();
	float total = 0.0;
	for (int kk = 0; kk < des_dim; ++kk)
	{
		total += fabs(sdes[kk]);
	}
	total = total < 0.00001 ? 1 : total;
	for (int kk = 0; kk < des_dim; ++kk)
		sdes[kk] = sdes[kk] < 0 ? -1 * sqrt(-1 * sdes[kk] / total) : sqrt(sdes[kk] / total);
}
/**
*此方法用于载入模型矩阵，格式第一行两个数字是行数和列数，都入后转化为float型
*@param filePath 矩阵路径及文件名
*@param ml_model 输出矩阵
*/
bool load_metric_model(string filePath, cv::Mat& ml_model,string method)
{

	ifstream inputF(filePath.c_str());
	if (!inputF.is_open())
		return false;
	string line;
	getline(inputF, line);
	vector<string> ww;
	split_words(line, " ", ww);
	int row = atoi(ww[0].c_str());
	int col = atoi(ww[1].c_str());
	cout << "MetricMat Info: " << row << "\t" << col << endl;

	cv::Mat ml_model_ = cv::Mat(row, col, CV_32F);
	for (int i = 0; i < row; i++)
	{
		getline(inputF, line);
		split_words(line, " ", ww);
		for (int j = 0; j < col; j++){
			ml_model_.at<float>(i, j) = atof(ww[j].c_str());
		}
	}
	if (method == "SP")
	{
		cv::transpose(ml_model_, ml_model);///SP方法需要将投影矩阵转置
	}
	else
	{
    	ml_model = ml_model_;
	}
	return true;
}
//TODO: 考虑把归一化单独拿出来
void do_metric( Mat& Model, Mat& data, Mat& result)
{
	///L2 Norm
	///L2NormFeature(smat);
	///reduction
#ifdef USE_OPENCV_MAT

	cv::gemm(data, Model, 1.0, 0.0, 0.0, result);
#else

	Eigen::Map<Matric_DDF> e_data(data.ptr<float>(), data.rows, data.cols);
	Eigen::Map<Matric_DDF> e_model(Model.ptr<float>(), Model.rows, Model.cols);

	Matric_DDF e_result = e_data*e_model;
	result = Mat(e_result.rows(), e_result.cols(), CV_32FC1, e_result.data());
#endif
	
}
Mat patchWiseSubtraction(Mat sImg)
{
	Mat dImg;
	// patch-wise mean subtraction
	cv::Scalar mean, stddev;
	cv::meanStdDev(sImg, mean, stddev);
	{
		cv::Mat *slice = new cv::Mat[sImg.channels()];
		cv::split(sImg, slice);
		for (int c = 0; c < sImg.channels(); ++c) {
			cv::subtract(slice[c], mean[c], slice[c]);
		}
		cv::merge(slice, sImg.channels(), dImg);
		delete[] slice;
	}

	return dImg;
}


 //patch-wise stddev division
Mat patchWiseStdDevDiv(Mat sImg)
{
	Mat dImg;
	// patch-wise stddev division
	cv::Scalar mean, stddev;
	cv::meanStdDev(sImg, mean, stddev);

	cv::Mat *slice = new cv::Mat[sImg.channels()];
	cv::split(sImg, slice);
	for (int c = 0; c < sImg.channels(); ++c) {
		slice[c] /= stddev[c];
	}
	cv::merge(slice, dImg.channels(), dImg);
	delete[] slice;

	return dImg;
}

