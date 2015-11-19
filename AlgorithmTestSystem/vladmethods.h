#ifndef VLADMETHODS_H
#include <vlad.h>
#include <fisher.h>
#include <kmeans.h>
#include <gmm.h>
#include <dsift.h>
#include <sift.h>
#include <vlad.h>"

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "utls.h"
namespace vlad{
	/**
	*�����ϸ�˹ģ����ز���
	*@param modelFile ����·�����ļ���
	*@param gmm GMMģ��
	*/
	void saveGmmModel(const char * modelFile, VlGMM * gmm);
	void loadGmmModel(const char * modelFile, VlGMM *& gmm, vl_size & dimension, vl_size & numClusters);
	void loadKmeansModel(const char * modelFile, float*& km, vl_size & dimension, vl_size & numClusters);
	void saveKmeansModel(const char * modelFile, VlKMeans * km);
	/**
	*���ڲ�������sift����
	*@param sImg ����ͼ��
	*@param descriptors ͼ���в���������
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
}
#endif // !VLADMETHODS_H