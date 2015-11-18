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

//Include the files from the vl
#include "utls.h"
#include "vladmethods.h"
using namespace std;
using namespace cv;
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
	//vlad::ExitTheSiftFeature(trainlistfile);
	//vlad::TrainVladModel();
	//vlad::TestVladModel(testlistfile);
	vector<float>a{ 1.0, 2.0, 4.0 };
	Mat M2 = Mat(3, 1, CV_32FC1,a.data());
	cout << M2 << endl;
	cout << "work has been down" << endl;
	getchar();
	return 0;
}






