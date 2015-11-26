#include "vladmethods.h"
#include <iostream>
#include <fstream>
#include "Config.h"
#include "cluteranalysis.hpp"
#include "localfeature.hpp"
//TODO: 省略中间数据的存储和读取实现直接的使用计算
using namespace std;
const static int NUMBER_OF_IMAGES_TO_TRAIN = 100;
const static int NUMBER_OF_IMAGES_TO_TEST = 100;
static string PATH_OF_WORK = "D:/E/work/dressplus/code/temp/";
static string PATH_OF_IMAGE = "D:/E/work/dressplus/code/data/fvtraindata/";
static string NAME_OF_FEATUREFILE = "vladfeature.txt";
const static int SELECT_NUM = 1000;
static int  FEA_DIM = 128;
static int NUMBER_OF_CLUSTERS = 256;
const static int NUMBER_OF_IMAGES_TO_PCA = 100;
//#define REDUCE_FEATURE
//#define USE_PCA
#define linux
namespace vlad{
	/**@brief make some configure works such as the paths
	*...
	*/
	void configure(int cluster_number, int feature_dimension)
	{
		FEA_DIM = feature_dimension;
		NUMBER_OF_CLUSTERS = cluster_number;

		const char ConfigFile[] = "Config.txt";
		Config configSettings(ConfigFile);
		PATH_OF_IMAGE = configSettings.Read("path_of_image", PATH_OF_IMAGE);
		PATH_OF_WORK = configSettings.Read("path_of_work", PATH_OF_WORK);
	}

	VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum)
	{
		/// load VLAD model
		vl_size dimension = -1;
		vl_size numClusters = -1;
		float* vlad_km = nullptr;
		ClusterAnaysis::loadKmeansModel(vlad_km_path.c_str(), vlad_km, dimension, numClusters);
		feature_dim = dimension*numClusters;
		VlKMeans *kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
		vl_kmeans_set_centers(kmeans, vlad_km, dimension, numClusters);
		vl_free(vlad_km);
		feature_dim = dimension;
		clusterNum = numClusters;
		return kmeans;
	}

	vector<float> encodeVladFea(VlKMeans *vladModel, vector<float> rawFea, int feature_dim, int clusterNum)
	{
		float * enc = (float*)vl_malloc(sizeof(float)*feature_dim*clusterNum);
		int elemSize = rawFea.size() / feature_dim;
		/// find nearest cluster centers for the data that should be encoded
		vl_uint32* idx_v = (vl_uint32*)vl_malloc(sizeof(vl_uint32) * elemSize);
		float* dists = (float*)vl_malloc(sizeof(float) * elemSize);
		vl_kmeans_quantize(vladModel, idx_v, dists, rawFea.data(), elemSize);
		///convert indexes array to assignments array,
		///which can be processed by vl_vlad_encode
		float* assignments = (float*)vl_malloc(sizeof(float) * elemSize * clusterNum);
		memset(assignments, 0, sizeof(float) * elemSize * clusterNum);
		for (int i = 0; i < elemSize; i++) {
			assignments[i * clusterNum + idx_v[i]] = 1.;
		}
		///do the encoding job
		vl_vlad_encode(enc, VL_TYPE_FLOAT,
			vl_kmeans_get_centers(vladModel), feature_dim, clusterNum,
			rawFea.data(), elemSize,
			assignments,
			0);

		//vector<float> vladFea(feature_dim*clusterNum, 0);
		vector<float> vladFea(enc, enc + feature_dim*clusterNum);///替代元素的复制环节
		//TODO: find a way to initial the vector directorly
		//memcpy(vladFea.data(), enc, feature_dim*clusterNum*sizeof(float));
		vl_free(dists);
		vl_free(assignments);
		vl_free(idx_v);
		vl_free(enc);
		return vladFea;
	}

	PCA getPCAmodel(string trainlistfile, int maxComponents = FEA_DIM)
	{
		ifstream inputtrainlistF(trainlistfile.c_str());
		string line;
		int cnt = 0;
		int img_number = 0;
		Mat descriptor_set;
		std::cout << "start process the train image to get the PCA model" << endl;
		while (getline(inputtrainlistF, line))
		{
			try{
				if (cnt++ % 100 == 0)
					std::cout << "proc " << cnt << endl;
				if (img_number > NUMBER_OF_IMAGES_TO_PCA)
				{
					break;
				}
				img_number++;
				srand((unsigned)getTickCount());

				Mat img = imread(PATH_OF_IMAGE + line, 0);
				if (img.empty() || img.cols < 64 || img.rows < 64)
					continue;
				double t = (double)cv::getTickCount();
				Mat descriptors;
				LocalFeature::extDenseVlSiftDes(img, descriptors);
				t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
				std::cout << t << " s" << std::endl;
				descriptor_set.push_back(descriptors);
			}

			catch (...){
				std::cout << "there are something wrong with picture" << PATH_OF_IMAGE << line << endl;
				continue;
			}
		}
		PCA pca(descriptor_set, Mat(), PCA::DATA_AS_ROW, maxComponents);
		FileStorage pcafile("data/pca_model.yml", FileStorage::WRITE);
		pcafile << "mean" << pca.mean;
		pcafile << "e_vectors" << pca.eigenvectors;
		pcafile << "e_values" << pca.eigenvalues;
		return pca;
	}
	void  loadPCAmodel(const string pcafilepath, PCA& pca)
	{
		FileStorage pcafile(pcafilepath, FileStorage::READ);
		if (!pcafile.isOpened())
		{
			std::cout << "can not opencv the pcafiel" << endl;
		}
		pcafile["mean"] >> pca.mean;
		pcafile["e_vectors"] >> pca.eigenvectors;
		pcafile["e_values"] >> pca.eigenvalues;

	}

	PCA compressPCA(const Mat& pcaset, int maxComponents,
		const Mat& testset, Mat& compressed)
	{
		PCA pca(pcaset, // pass the data
			Mat(), // we do not have a pre-computed mean vector,
			// so let the PCA engine to compute it
			PCA::DATA_AS_ROW, // indicate that the vectors
			// are stored as matrix rows
			// (use PCA::DATA_AS_COL if the vectors are
			// the matrix columns)
			maxComponents // specify, how many principal components to retain
			);
		// if there is no test data, just return the computed basis, ready-to-use
		if (!testset.data)
			return pca;
		CV_Assert(testset.cols == pcaset.cols);
		compressed.create(testset.rows, maxComponents, testset.type());
		pca.project(testset, compressed);
		FileStorage pcafile("pca_model.yml", FileStorage::WRITE);
		pca.write(pcafile);
		/*Mat reconstructed;
		for (int i = 0; i < testset.rows; i++)
		{
		Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
		// compress the vector, the result will be stored
		// in the i-th row of the output matrix
		pca.project(vec, coeffs);
		// and then reconstruct it
		//pca.backProject(coeffs, reconstructed);
		// and measure the error
		//printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
		}*/
	}

	void ExitTheSiftFeature(string trainlistfile)
	{
		ifstream inputtrainlistF(trainlistfile.c_str());
#ifdef REDUCE_FEATURE
		// load model 
#ifdef USE_PCA
		PCA pca;
		loadPCAmodel("data/pca_model.yml", pca);
#else
		Mat mlModel;
		try{
			load_metric_model(PATH_OF_WORK + "DimentionReduceMat_vlsift_32.txt", mlModel, "SP");
		}
		catch (...){
			std::cout << "can not load the reduce matrix" << endl;
		}
#endif
#endif
		ofstream outputF("vlsift_tmp.fea");
		string imagename;
		int cnt = 0;
		int img_number = 0;
		Mat descriptor_set;
		vector<string> trainlist;
		while (getline(inputtrainlistF, imagename))
			trainlist.push_back(imagename);
#ifdef linux
#pragma omp parallel for


		for (string line : trainlist)
		{
#else // linux
		for (string line : trainlist)
		{
#endif

			try{
				if (cnt++ % 100 == 0)
					cout << "proc " << cnt << endl;
				if (img_number > NUMBER_OF_IMAGES_TO_TRAIN)
				{
					break;
				}
				img_number++;
				srand((unsigned)getTickCount());

				Mat img = imread(PATH_OF_IMAGE + line, 0);
				if (img.empty() || img.cols < 64 || img.rows < 64)
					continue;
				double t = (double)cv::getTickCount();
				Mat descriptors;
				LocalFeature::extDenseVlSiftDes(img, descriptors);
			    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
				std::cout << t << " s" << std::endl;
#ifdef REDUCE_FEATURE
#ifdef USE_PCA
				Mat reduced_descriptors = pca.project(descriptors);
				vector<float> normfea = getVector(reduced_descriptors);

#else
				vector<float> normfea = LocalFeature::genDescriptorReduced(descriptors, mlModel);
#endif
#else
				vector<float> normfea = getVector(descriptors);
#endif
				cout << descriptors.rows << endl;
				int len = normfea.size() / FEA_DIM;
				assert(normfea.size() % FEA_DIM == 0);
				for (int j = 0; j < min(SELECT_NUM, len); j++)
				{
					int beg = (rand() % len)*FEA_DIM;
					for (int i = beg; i < beg + FEA_DIM; i++)
						outputF << normfea[i] << " ";

					outputF << endl;
				}
			}
			catch (...){
				cout << "there are something wrong with picture" << PATH_OF_IMAGE << line << endl;
				continue;
			}
		}
		cout << "proc " << cnt << endl;
		inputtrainlistF.close();
		outputF.close();
		}
	//TODO: this function actually is a keams training function so I can move it to clusteranaysis.hpp if needed
	VlKMeans* getKmeansModel(int cluster_num, int feature_dim)
	{
		// set params
		vl_set_num_threads(0); /* use the default number of threads */

		vl_size dimension = feature_dim;
		vl_size numClusters = cluster_num;
		vl_size maxiter = 128;
		vl_size maxrep = 1;

		//	VlKMeans * kmeans = 0;
		vl_size maxiterKM = 64;
		vl_size ntrees = 3;
		vl_size maxComp = 64;

		cout << endl;
		cout << "Encode params: dimension->" << dimension << endl;
		cout << "Encode params: numCluster->" << numClusters << endl;
		cout << endl;

		// init kmeans status
		VlKMeans* kmeans;
		kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
		vl_kmeans_set_verbosity(kmeans, 1);
		vl_kmeans_set_max_num_iterations(kmeans, maxiterKM);
		vl_kmeans_set_max_num_comparisons(kmeans, maxComp);
		vl_kmeans_set_num_trees(kmeans, ntrees);
		vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
		vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);


		string kmeansF = "vlad128_sift32.model";
		ifstream inputF("vlsift_tmp.fea");
		string line;
		vector<float> data_des;
		while (getline(inputF, line))
		{
			vector<string> ww;
			split_words(line, " ", ww);
			assert(ww.size() == FEA_DIM);
			for (int i = 0; i < ww.size(); i++)
				//TODO: bad alloc
				data_des.push_back(atof(ww[i].c_str()));
		}


		cout << "Invoke VLAD" << endl;
		// create a KMeans object and run clustering to get vocabulary words (centers)
		vl_kmeans_cluster(kmeans,
			data_des.data(),
			dimension,
			data_des.size() / dimension,
			numClusters);
		cout << "Train kmeans model successful" << endl;
		ClusterAnaysis::saveKmeansModel(kmeansF.c_str(), kmeans);
		cout << "Save model to " << kmeansF << endl;
		return kmeans;
	}
#ifdef REDUCE_FEATURE
#ifdef USE_PCA
	 vector<float> VladFeatureEncode(Mat& img, vl_size &dimension, vl_size &numClusters, VlKMeans* kmeans, const PCA&pca)
#else
	 vector<float> VladFeatureEncode(Mat& img, vl_size &dimension, vl_size &numClusters, VlKMeans* kmeans, const Mat& mlModel)
#endif
#else
      vector<float> VladFeatureEncode(Mat& img, vl_size &dimension, vl_size &numClusters, VlKMeans* kmeans)
#endif
	{
		double t = (double)cv::getTickCount();
		Mat descriptors;
		//extDenseVlSiftDes(img, descriptors);
		LocalFeature::extSparseVlSiftDes(img, descriptors);
#ifdef REDUCE_FEATURE
#ifdef USE_PCA
		Mat reduced_descriptors = pca.project(descriptors);
		vector<float> normfea = getVector(reduced_descriptors);

#else
		vector<float> normfea = LocalFeature::genDescriptorReduced(descriptors, mlModel);
#endif
#else
		std::vector<float> normfea = getVector(descriptors);
#endif
		int len = normfea.size() / dimension;
		assert(normfea.size() % dimension == 0);

		vector<float> vlf = vlad::encodeVladFea(kmeans, normfea, dimension, numClusters);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << t << " s" << std::endl;

		assert(dimension*numClusters == vlf.size());
		return vlf;
	}


	void  GetVladFeature(string testlistfile)
	{

		ifstream inputF(testlistfile.c_str());
#ifdef REDUCE_FEATURE
		// load model 
#ifdef USE_PCA
		PCA pca;
		loadPCAmodel("data/pca_model.yml", pca);
#else
		Mat mlModel;
		bool flag = load_metric_model(PATH_OF_WORK + "DimentionReduceMat_vlsift_32.txt", mlModel, "SP");
		//TODO: add a assert
#endif
#endif
		//load VLAD model
		//vl_size dimension = -1;
		//vl_size numClusters = -1;
		//VlKMeans *kmeans = vlad::initVladEngine("vlad128_sift32.model", dimension, numClusters);
		//VlKMeans *kmeans = new VlKMeans();
		VlKMeans*kmeans = getKmeansModel(NUMBER_OF_CLUSTERS, FEA_DIM);
		vl_size dimension = vl_kmeans_get_dimension(kmeans);
		vl_size numClusters = vl_kmeans_get_num_centers(kmeans);
		//------
		ofstream outputF(NAME_OF_FEATUREFILE);
		string imagename;
		int cnt = 0;
		int img_number = 0;
		vector<string> trainlist;
		while (getline(inputF, imagename))
			trainlist.push_back(imagename);
#ifdef linux
		//#pragma omp parallel for
		for (string line : trainlist)//^M
		{//^M
#else // linux
		for (string line : trainlist)
		{
#endif
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
#ifdef REDUCE_FEATURE
#ifdef USE_PCA
				vector<float>vlf = VladFeatureEncode(img, dimension, numClusters, kmeans, pca);
#else
				vector<float>vlf = VladFeatureEncode(img, dimension, numClusters, kmeans, mlModel);
#endif
#else
				vector<float>vlf = VladFeatureEncode(img, dimension, numClusters, kmeans);
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
		vl_kmeans_delete(kmeans);
		inputF.close();
		outputF.close();
		}
	}