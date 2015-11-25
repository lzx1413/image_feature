#ifndef UTILS_H
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;
void L2NormFeature(Mat& smat, int rowidx);
void L2NormFeature(Mat& smat);
void split_words(const string& src, const string& separator, vector<string>& dest);
void RootNormFeature(InputOutputArray& sdes);
bool load_metric_model(string filePath, cv::Mat& ml_model, string method);
void do_metric(Mat& mlModel, InputArray& smat, Mat& dmat);
vector<float> getVector(const Mat &_t1f);
#endif // !UTILS_H