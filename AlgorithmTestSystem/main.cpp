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
	int cluster_num = 512;
	int feature_dimention = 32;
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
		else if (string(argv[i]) == "--clusternum")
		{
			cluster_num = std::stoi(argv[++i]);
		}
		else if (string(argv[i]) == "--featuredim")
		{
			feature_dimention = std::stoi(argv[++i]);
		}

	}
	    //vlad::configure(cluster_num,feature_dimention);
		//vlad::getPCAmodel(trainlistfile,32);
    	//vlad::ExitTheSiftFeature(trainlistfile);	
	   // vlad::TestVladModel(testlistfile);
	vector<float> feature{ 1, 1, 1, 1, 1, 1 };
	RootNormFeature(feature);
	Mat a = Mat::ones(4, 6,CV_32FC1);
	for (float a:feature)
	{
		cout << a << endl;
	}
	a.at<float>(0, 0) = 0;
	RootNormFeature(a);
	cout << a;
	getchar();
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
	//MethodTimeResume timetest("time.log");
	//timetest.test();
	//getchar();

	//vlad::getPCAmodel(trainlistfile, 32);
	
}






