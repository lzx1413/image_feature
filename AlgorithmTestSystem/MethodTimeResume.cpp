#include "MethodTimeResume.h"
#include "StopWatch.hpp"
#include "utls.h"
#include <opencv2/core/core.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core.hpp"
#include "Config.h"

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
	int port;
	std::string ipAddress;
	std::string username;
	std::string password;
	const char ConfigFile[] = "Config.txt";
	Config configSettings(ConfigFile);

	port = configSettings.Read("port", 0);
	ipAddress = configSettings.Read("ipAddress", ipAddress);
	username = configSettings.Read("username", username);
	password = configSettings.Read("password", password);
	std::cout << "port:" << port << std::endl;
	std::cout << "ipAddress:" << ipAddress << std::endl;
	std::cout << "username:" << username << std::endl;
	std::cout << "password:" << password << std::endl;
}

void   MethodTimeResume::testMatrixMulti()
{
	
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

