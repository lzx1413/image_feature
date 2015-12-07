#ifndef CLUSTERANALYSIS_HPP
#define CLUSTERANALYSIS_HPP
#include <vl/vlad.h>
#include <vl/fisher.h>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/dsift.h>
#include <vl/sift.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "utls.h"
class ClusterAnaysis{
public:
	static void saveGmmModel(const char * modelFile, VlGMM * gmm)
	{
		vl_size d, cIdx;
		vl_size dimension = vl_gmm_get_dimension(gmm);
		vl_size numClusters = vl_gmm_get_num_clusters(gmm);
		float * sigmas = (float*)vl_gmm_get_covariances(gmm);
		float * means = (float*)vl_gmm_get_means(gmm);
		float * weights = (float*)vl_gmm_get_priors(gmm);
		ofstream ofp(modelFile);
		ofp << dimension << " " << numClusters << endl;
		for (cIdx = 0; cIdx < numClusters; cIdx++) {
			for (d = 0; d < dimension; d++)
				ofp << means[cIdx*dimension + d] << " ";
			for (d = 0; d < dimension; d++)
				ofp << sigmas[cIdx*dimension + d] << " ";
			ofp << weights[cIdx] << endl;
		}
		ofp.close();
	}


	static void loadGmmModel(const char * modelFile, VlGMM *& gmm, vl_size & dimension, vl_size & numClusters)
	{
		vl_size d, cIdx;

		ifstream ifp(modelFile);
		if (!ifp.is_open())
		{
			std::cout << "Can't open " << modelFile << endl;
			exit(-1);
		}
		string line;
		getline(ifp, line);
		vector<string> ww;
		split_words(line, " ", ww);
		dimension = atoi(ww[0].c_str());
		numClusters = atoi(ww[1].c_str());

		gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

		float * sigmas = (float*)vl_malloc(sizeof(float)*dimension*numClusters);
		float * means = (float*)vl_malloc(sizeof(float)*dimension*numClusters);
		float * weights = (float*)vl_malloc(sizeof(float)*numClusters);

		for (cIdx = 0; cIdx < numClusters; cIdx++) {
			getline(ifp, line);
			ww.clear();
			split_words(line, " ", ww);
			int xIdx = 0;
			for (d = 0; d < dimension; d++){
				means[cIdx*dimension + d] = atof(ww[xIdx].c_str());
				xIdx++;
			}
			for (d = 0; d < dimension; d++){
				sigmas[cIdx*dimension + d] = atof(ww[xIdx].c_str());
				xIdx++;
			}
			weights[cIdx] = atof(ww[xIdx].c_str());
		}
		vl_gmm_set_means(gmm, means);
		vl_gmm_set_covariances(gmm, sigmas);
		vl_gmm_set_priors(gmm, weights);

		ifp.close();

		vl_free(sigmas);
		vl_free(means);
		vl_free(weights);

		cout << "Load model ok " << endl;
	}

	static void saveKmeansModel(const char * modelFile, VlKMeans * km)
	{
		vl_size dimension = vl_kmeans_get_dimension(km);
		vl_size numClusters = vl_kmeans_get_num_centers(km);
		float * centers = (float*)vl_kmeans_get_centers(km);

		ofstream ofp(modelFile);
		ofp << dimension << " " << numClusters << endl;
		for (int cIdx = 0; cIdx < numClusters*dimension; cIdx++) {
			ofp << centers[cIdx] << " ";
		}

		ofp.close();
	}


	static void loadKmeansModel(const char * modelFile, float*& km, vl_size & dimension, vl_size & numClusters)
	{
		ifstream ifp(modelFile);
		if (!ifp.is_open())
		{
			cout << "Can't open " << modelFile << endl;
			exit(-1);
		}
		string line;
		getline(ifp, line);
		vector<string> ww;
		split_words(line, " ", ww);
		dimension = atoi(ww[0].c_str());
		numClusters = atoi(ww[1].c_str());

		km = (float*)vl_malloc(sizeof(float)*dimension*numClusters);
		getline(ifp, line);
		ww.clear();
		split_words(line, " ", ww);
		for (int cIdx = 0; cIdx < numClusters*dimension; cIdx++) {
			km[cIdx] = atof(ww[cIdx].c_str());
		}
		ifp.close();
		cout << "Load model ok " << endl;
	}
	static void getSURFwithGPU()
	{

	}
};


#endif // !CLUSTERANALYSIS_HPP
