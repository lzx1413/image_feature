#ifndef VLADMETHODS_H
#define VLADMETHODS_H
#include <vl/vlad.h>
#include <vl/fisher.h>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/dsift.h>
#include <vl/sift.h>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "utls.h"
namespace vlad{
	void configure(int clusternum,int featuredim);
	PCA compressPCA(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressed);
	//INIT VLAD ENGINE
	VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum);
	vector<float> encodeVladFea(VlKMeans *vladModel, vector<float> rawFea, int feature_dim, int clusterNum);
	void ExitTheSiftFeature(string trainlistfile);
	void TrainVladModel();
	void TestVladModel(string testlistfile);
	PCA getPCAmodel(string trainlistfile, int maxComponents);
}
#endif // !VLADMETHODS_H