#ifndef VLADMETHODS_H
#include <vl/vlad.h>
#include <vl/fisher.h>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/dsift.h>
#include <vl/sift.h>
#include <vl/vlad.h>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "utls.h"
namespace vlad{
	void configure();
	/**@brief 保存混合高斯模型相关参数
	*@param modelFile 保存路径及文件名
	*@param gmm GMM模型
	*/
	void saveGmmModel(const char * modelFile, VlGMM * gmm);
	void loadGmmModel(const char * modelFile, VlGMM *& gmm, vl_size & dimension, vl_size & numClusters);
	void loadKmeansModel(const char * modelFile, float*& km, vl_size & dimension, vl_size & numClusters);
	void saveKmeansModel(const char * modelFile, VlKMeans * km);
	/**@brief 用于产生大量sift特征
	*@param sImg 输入图像
	*@param descriptors 图像中产生的特征
	*/
	void extDenseVlSiftDes(Mat& sImg, Mat& descriptors);
	vector<float> getVector(const Mat &_t1f);
	vector<float> genDescriptorReduced(Mat& descriptors, Mat& mlModel);
	void extSparseVlSiftDes(Mat &sImg, Mat& descriptors);
	PCA compressPCA(const Mat& pcaset, int maxComponents, const Mat& testset, Mat& compressed);
	//INIT VLAD ENGINE
	VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum);
	vector<float> encodeVladFea(VlKMeans *vladModel, vector<float> rawFea, int feature_dim, int clusterNum);
	void ExitTheSiftFeature(string trainlistfile);
	void TrainVladModel();
	int TestVladModel(string testlistfile);
	PCA getPCAmodel(string trainlistfile, int maxComponents);
}
#endif // !VLADMETHODS_H