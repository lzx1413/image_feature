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
#include "fishermethods.hpp"
#include "StopWatch.hpp"
#include "testpca.hpp"
using namespace std;
using namespace cv;
using Eigen::MatrixXd;
const static int LENGTH_OF_SINGLE_DATA = 128;
//#define ReduceMatrix
//#define VLAD
//#define kmeans
//#define kmeans_pca
#define test_pca
void help()
{
	cout << "Ussage:--trainlist <trianlist \n\t --testlist <testlist>" << endl;
}
int main(int argc, char* argv[])
{
	string trainlistfile;
	string testlistfile;
	string modelfile;
	string rawfeaturefile;
	string resultpath;
	string kmeansfilepath;
	int maxComponent = 32;
	int cluster_num = 512;
	int feature_dimention = 128;
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
		else if (string(argv[i]) == "--modelfile")
		{
			modelfile = argv[++i];
		}
		else if (string(argv[i]) == "--rawfeature")
		{
			rawfeaturefile = argv[++i];
		}
		else if (string(argv[i])=="--resultpath")
		{
			resultpath = argv[++i];

		}
		else if (string(argv[i]) == "--maxComponent")
		{
			maxComponent = std::stoi(argv[++i]);

	}
		else if (string(argv[i]) == "--kmeansfilepath")
		{
			kmeansfilepath = argv[++i];

		}
	}
#ifdef test_pca
	Mat img1 = imread("data/img1.png");
	Mat img2 = imread("data/img2.png");
	if (img1.empty()||img2.empty())
	{
		cout << "can not open imags" << endl;
	}
	testPCA(img1, img2);
#endif
#ifdef kmeans_pca

#endif
#ifdef kmeans
	vlad::getKmeansModel(cluster_num,feature_dimention,rawfeaturefile,resultpath);
#endif
#ifdef VLAD
	Stopwatch watch;
	watch.Start();
	vlad::GetVladFeatureFromSift(kmeansfilepath,testlistfile);
	watch.Stop();
	cout<<"getkmeans use "<<watch.GetTime()<<endl;
	getchar();
#endif
#ifdef ReduceMatrix
	ifstream rawfeature(rawfeaturefile.c_str());
	ofstream outPut(resultpath.c_str());
	string single_raw_fea;
	PCA pca;
	vlad::loadPCAmodel(modelfile, pca);
	while (getline(rawfeature,single_raw_fea))
	{
		vector<string> ww;
		cv::Mat rawfeature_mat = cv::Mat(1,LENGTH_OF_SINGLE_DATA, CV_32F);
		split_words(single_raw_fea, " ", ww);
		for (int j = 0; j < LENGTH_OF_SINGLE_DATA; ++j){
			rawfeature_mat.at<float>(0, j) = atof(ww[j].c_str());
		}
		cv::Mat matric_reduced;
		matric_reduced = pca.project(rawfeature_mat);
		for (int i=0;i<matric_reduced.cols;++i)
		{
			outPut<<matric_reduced.at<float>(0,i);
		}
		outPut <<endl;
        
	}
	rawfeature.close();
	outPut.close();
#else
#ifdef ALL



	    vlad::configure(cluster_num,feature_dimention);
		//vlad::getPCAmodel(trainlistfile,32);
    	vlad::ExitTheSiftFeature(trainlistfile);	
	   // vlad::GetVladFeature(testlistfile);
//	FV::GetGMMModel(32, 512);
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
#endif
#endif // ReduceMatrix
	
}






