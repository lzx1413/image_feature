#include "MethodTimeResume.h"
#include"StopWatch.hpp"

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
}
void MethodTimeResume::testEncode2Binary()
{
	Mat data = Mat(100000, 1024, CV_32FC1);
	randu(data, -10, 10);
	Mat model = Mat(1024, 512, CV_32FC1);
	randu(model, -1, 1);
	Mat result;
	Stopwatch time;
	time.Start();
	encode2Binary(data, model, result);
	time.Stop();
	auto time_resume = time.GetTime();
	time.Reset();
	cout << time_resume << endl;
	time_log << "opencv Mat function 100000*1024 1024*512 time is " << time_resume << endl;
}

