#ifndef FISHERMETHOD_H
#define FISHERMETHOD_H
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vl/vlad.h>
#include <vl/fisher.h>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/dsift.h>
#include <vl/vlad.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include "utls.h"
#include "cluteranalysis.hpp"
#include "localfeature.hpp"

const static int NUMBER_OF_IMAGES_TO_TRAIN = 100;
const static int NUMBER_OF_IMAGES_TO_TEST = 100;
static string PATH_OF_WORK = "D:/E/work/dressplus/code/temp/";
static string PATH_OF_IMAGE = "D:/E/work/dressplus/code/data/fvtraindata/";
static string NAME_OF_FEATUREFILE = "VFfeature.txt";
static int  FEA_DIM = 32;
static int NUMBER_OF_CLUSTERS = 512;
class FV{
	static void configure()
	{

	}
	static VlGMM* GetGMMModel(int dimension_, int numclusters_)
	{
		vector<float> data_des;
		ifstream inputFea("vlsift_tmp.fea");
		if (!inputFea.is_open())
			cout << "can not open the data file" << endl;

		string line;
		while (getline(inputFea, line))
		{
			vector<string> ww;
			split_words(line, " ", ww);
			for (int i = 0; i < ww.size(); i++)
				data_des.push_back(atof(ww[i].c_str()));
		}
		inputFea.close();

		vl_size dimension = dimension_;
		vl_size numClusters = numclusters_;
		vl_size maxiter = 250;
		vl_size maxrep = 1;

		VlKMeans * kmeans = 0;
		vl_size maxiterKM = 250;
		vl_size ntrees = 3;
		vl_size maxComp = 120;

		cout << endl;
		cout << "Encode params: dimension->" << dimension << endl;
		cout << "Encode params: numCluster->" << numClusters << endl;
		cout << endl;

		// init kmeans status
		kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
		vl_kmeans_set_verbosity(kmeans, 1);
		vl_kmeans_set_max_num_iterations(kmeans, maxiterKM);
		vl_kmeans_set_max_num_comparisons(kmeans, maxComp);
		vl_kmeans_set_num_trees(kmeans, ntrees);
		vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
		vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);
		vl_size dataIdx, cIdx;
		VlGMM * gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);
		double sigmaLowerBound = 0.000001;

		vl_gmm_set_initialization(gmm, VlGMMKMeans);
		vl_gmm_set_kmeans_init_object(gmm, kmeans);


		// init gmm status
		vl_gmm_set_max_num_iterations(gmm, maxiter);
		vl_gmm_set_num_repetitions(gmm, maxrep);
		vl_gmm_set_verbosity(gmm, 1);
		vl_gmm_set_covariance_lower_bound(gmm, sigmaLowerBound);

		// run gmm core
		vl_gmm_cluster(gmm, data_des.data(), data_des.size() / dimension);
		cout << "Train gmm model successful" << endl;
		ClusterAnaysis::saveGmmModel("gmmmodel.model", gmm);

	}

	static vector<float> encodeFVfeature(VlGMM*gmm, vector<float>&normfea, vl_size numClusters, vl_size dimension)
	{
		if (normfea.size() > 0)
		{
			int FVfeature_dim = dimension*numClusters * 2;
			float * enc = (float*)vl_malloc(sizeof(float)*dimension);
			vl_fisher_encode(enc, VL_TYPE_FLOAT,
				vl_gmm_get_means(gmm), dimension, numClusters,
				vl_gmm_get_covariances(gmm),
				vl_gmm_get_priors(gmm),
				normfea.data(), normfea.size() / dimension,
				VL_FISHER_FLAG_IMPROVED
				);
			vector<float>VFfeature(enc, enc + FVfeature_dim);///替代元素的复制环节
			//memcpy(vladFea.data(), enc, feature_dim*clusterNum*sizeof(float));
			vl_free(enc);
			return VFfeature;
		}
	};
#ifdef USE_PCA
	vector<float> FVFeatureEncode(Mat& img, vl_size &dimension, vl_size &numClusters, VlGMM* gmm, const PCA&pca)
#else
	vector<float> FVFeatureEncode(Mat& img, vl_size &dimension, vl_size &numClusters, VlGMM* gmm, const Mat& mlModel)
#endif
	{
		double t = (double)cv::getTickCount();
		Mat descriptors;
		//extDenseVlSiftDes(img, descriptors);
		LocalFeature::extSparseVlSiftDes(img, descriptors);
#ifdef USE_PCA
		Mat reduced_descriptors = pca.project(descriptors);
		vector<float> normfea = getVector(reduced_descriptors);

#else
		vector<float> normfea = LocalFeature::genDescriptorReduced(descriptors, mlModel);
#endif
		int len = normfea.size() / dimension;
		assert(normfea.size() % dimension == 0);

		vector<float> vlf = FV::encodeFVfeature(gmm, normfea, dimension, numClusters);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << t << " s" << std::endl;

		assert(dimension*numClusters * 2 == vlf.size());
		return vlf;
	}

	static void  GetFVFeature(string testlistfile)
	{

		ifstream inputF(testlistfile.c_str());
		// load model 
#ifdef USE_PCA
		PCA pca;
		loadPCAmodel("data/pca_model.yml", pca);
#else
		Mat mlModel;
		bool flag = load_metric_model(PATH_OF_WORK + "DimentionReduceMat_vlsift_32.txt", mlModel, "SP");
		//TODO: add a assert
#endif
		//load VLAD model
		//vl_size dimension = -1;
		//vl_size numClusters = -1;
		//VlKMeans *kmeans = vlad::initVladEngine("vlad128_sift32.model", dimension, numClusters);
		//VlKMeans *kmeans = new VlKMeans();
		VlGMM*gmm = GetGMMModel(NUMBER_OF_CLUSTERS, FEA_DIM);
		vl_size dimension = vl_gmm_get_dimension(gmm);
		vl_size numClusters = vl_gmm_get_num_clusters(gmm);
		//------
		ofstream outputF(NAME_OF_FEATUREFILE);
		string imagename;
		int cnt = 0;
		int img_number = 0;
		vector<string> trainlist;
		while (getline(inputF, imagename))
			trainlist.push_back(imagename);
#ifdef linux
#pragma omp parallel for
		for (string line : trainlist) ^ M
		{ ^M
#else // linux
		for (string line : trainlist)
		{
#endif{
			if (cnt++ % 100 == 0)
				cout << "proc " << cnt << endl;
			if (img_number > NUMBER_OF_IMAGES_TO_TEST)
			{
				//break;
			}
			try{
				img_number++;
				Mat imgS = imread(PATH_OF_IMAGE + line, 0);
				if (imgS.empty() || imgS.cols < 64 || imgS.rows < 64)
					continue;
				// norm image
				int normWidth = 360;
				int normHeight = 360.0 / imgS.size().width*imgS.size().height;
				Mat img;
				resize(imgS, img, Size(normWidth, normHeight));
#ifdef USE_PCA
				vector<float>vlf = FVFeatureEncode(img, dimension, numClusters, gmm, pca);
#else
				vector<float>vlf = FVFeatureEncode(img, dimension, numClusters, gmm, mlModel);
#endif
				outputF << line;
				for (int i = 0; i < vlf.size(); i++)
					outputF << " " << vlf[i];
				outputF << endl;
			}
			catch (...){
				cout << "there are something wrong with picture" << PATH_OF_IMAGE << line << endl;
				continue;
			}
		}
		vl_gmm_delete(gmm);
		inputF.close();
		outputF.close();
		}
	}
};
#endif