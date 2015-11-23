// TestVLAD_FV.cpp : 定义控制台应用程序的入口点。
//
//include the files form opencv

//stl
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <ctime>
#include <Eigen/Eigen>
#include "utls.h"
#include "vladmethods.h"
#include "MethodTimeResume.h"
using namespace std;
using namespace cv;
using Eigen::MatrixXd;
void help()
{
	cout << "Ussage:--trainlist <trianlist \n\t --testlist <testlist>" << endl;
}
int main(int argc, char* argv[])
{
	string trainlistfile;
	string testlistfile;
	if (argc == 1)
	{
		help();
		return -1;
	}
	for (int i = 1; i < argc;++i)
	{
		
		if (string(argv[i])=="--trainlist")
		{
			trainlistfile = argv[++i];
		}
		else if (string(argv[i]) == "--testlist")
		{
			testlistfile = argv[++i];
		}

	}
	    vlad::configure();
    	vlad::ExitTheSiftFeature(trainlistfile);
	///vlad::TrainVladModel();
	//vlad::TestVladModel(testlistfile);
    
	/*Mat rawdata = Mat(1, 3, CV_32FC1);
	rawdata.setTo(1);
	cout << rawdata << endl;
	Mat result;
	Mat model = (Mat_<float>(3, 2) << 1, 1, 1, -1, -1, -1);
	encode2Binary(rawdata, model, result);
	cout << "work has been down"<< rawdata << endl << model << endl << result << endl;
	vector<float> rawdata2{ 1.0, 1.0, 1.0 };
	Mat result2;
	encode2Binary(rawdata2, model, result2);
	cout << result2 << endl;
	Mat model2 = (Mat_<uchar>(3, 2) << 1, 1, 1, 1, 1, 1);
	model2.assignTo(model2, CV_32FC1);
	cout << model2<<endl<<model2.type();
	for (int i = 0; i < 10;i++)
	{
		for (int i = 0; i < 10;i++)
		{
			cout << i << endl;
		}
	}*/
	MethodTimeResume timetest("time.log");
	timetest.test();
	getchar();

	//vlad::getPCAmodel(trainlistfile, 32);
	
}






