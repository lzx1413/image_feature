#ifndef UTILS_H
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;
void L2NormFeature(Mat& smat, int rowidx);
void L2NormFeature(Mat& smat);
void split_words(const string& src, const string& separator, vector<string>& dest);
void RootNormFeature(vector<float>& sdes);
bool load_metric_model(string filePath, cv::Mat& ml_model, string method);
void do_metric( const Mat& mlModel, Mat& smat, Mat& dmat);
#endif // !UTILS_H