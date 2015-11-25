#ifndef VLADMETHODS_H
#define VLADMETHODS_H
#include <vl/kmeans.h>
#include <opencv2/opencv.hpp>

#include "utls.h"
namespace vlad{
	void configure(int clusternum,int featuredim);
	PCA compressPCA(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressed);
	//INIT VLAD ENGINE
	VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum);
	vector<float> encodeVladFea(VlKMeans *vladModel, vector<float> rawFea, int feature_dim, int clusterNum);
	void ExitTheSiftFeature(string trainlistfile);
	VlKMeans* getKmeansModel(int cluster_num, int feature_dim);
	void TestVladModel(string testlistfile);
	PCA getPCAmodel(string trainlistfile, int maxComponents);
}
#endif // !VLADMETHODS_H