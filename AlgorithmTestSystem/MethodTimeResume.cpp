#include "MethodTimeResume.h"
#include"StopWatch.hpp"
#include "utls.h"
#include <Eigen/Dense>
#include <opencv2/cvconfig.h>
#include <opencv2/core/core.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace Eigen;
MethodTimeResume::MethodTimeResume(string timefilelog) :time_log_file(timefilelog)
{
	time_log.open(timefilelog, ios::app|ios::out);
}


MethodTimeResume::~MethodTimeResume()
{
	time_log.close();
}
void MethodTimeResume::test()
{
	testEncode2Binary();
	testMatrixMulti();
}

void   MethodTimeResume::testMatrixMulti()
{
	std::srand((unsigned int)time(0));
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>data = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(1, 50000);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> model = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(50000, 1024);
	Stopwatch time;
	time.Start();
	for (int i = 0; i < 100; ++i)
	{
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>result = data*model;
	}
	time.Stop();
	cout << "eigen" << time.GetTime()<<endl;
	time.Reset();
	//cout << result << endl;
}
void MethodTimeResume::testEncode2Binary()
{
	Mat data = Mat(1, 50000, CV_32FC1);
	randu(data, -1, 1);
	Mat model = Mat(50000,1024, CV_32FC1);
	randu(model, -1, 1);
	Mat result;
	Stopwatch time;
	time.Start();
	for (int i = 0; i < 100;++i)
	{
		do_metric(model, data, result);
	}
	time.Stop();
	auto time_resume = time.GetTime();
	time.Reset();
	cout << "opencv"<<time_resume << endl;
	time_log << "opencv Mat function 100000*1024 1024*512 time is " << time_resume << endl;
}

