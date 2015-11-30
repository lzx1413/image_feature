#include "vladmethods.h"
#include <iostream>
#include <fstream>
#include "Config.h"
#include "cluteranalysis.hpp"
#include "localfeature.hpp"
#include "omp.h"
//TODO: 省略中间数据的存储和读取实现直接的使用计算
using namespace std;
const static int NUMBER_OF_IMAGES_TO_TRAIN = 10;
const static int NUMBER_OF_IMAGES_TO_TEST = 100;
static string PATH_OF_WORK = "D:/E/work/dressplus/code/temp/";
static string PATH_OF_IMAGE = "D:/E/work/dressplus/code/data/fvtraindata/";
static string NAME_OF_FEATUREFILE = "vladfeature.txt";
static string PATH_OF_SIFTFEATURE = "D:/E/work/dressplus/code/AlgorithmTestSystem/AlgorithmTestSystem/data/";
static string PATH_OF_VLADFEATURE = "";
const static int SELECT_NUM = 1000;
static int  FEA_DIM = 128;
static int NUMBER_OF_CLUSTERS = 256;
const static int NUMBER_OF_IMAGES_TO_PCA = 100;
//#define REDUCE_FEATURE
//#define USE_PCA
//#define linux
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
		L2NormFeature(vladFea);
		vl_free(dists);
		vl_free(assignments);
		vl_free(idx_v);
		vl_free(enc);
		return vladFea;
	}
	PCA getPCAmodel(string trainlistfile, int maxComponents, string resultpath)
	{
		//string kmeansF = "vlad_kmeans.model";
		ifstream inputF(trainlistfile);
		string line;
		vector<float> data_des;
		Mat feature_set;
		int count = 0;
		while (getline(inputF, line))
		{
	
			if (10 == count)
			{
				vector<string> ww;
				split_words(line, " ", ww);
				assert(ww.size() == FEA_DIM);
				for (int i = 0; i < ww.size(); i++)
					//TODO: bad alloc
				data_des.push_back(atof(ww[i].c_str()));
				count=0;
			    Mat single_fea = Mat(1, data_des.size(), CV_32FC1, data_des.data());
				feature_set.push_back(single_fea);
				data_des.clear();
			}
		
			count++;
		}
		PCA pca(feature_set, Mat(), PCA::DATA_AS_ROW, maxComponents);
		FileStorage pcafile(resultpath, FileStorage::WRITE);
		pcafile << "mean" << pca.mean;
		pcafile << "e_vectors" << pca.eigenvectors;
		pcafile << "e_values" << pca.eigenvalues;
		return pca;

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
		
			maxComponents // specify, how many principal components to retain
			);
		if (!testset.data)
			return pca;
		CV_Assert(testset.cols == pcaset.cols);
		compressed.create(testset.rows, maxComponents, testset.type());
		pca.project(testset, compressed);
		FileStorage pcafile("pca_model.yml", FileStorage::WRITE);
		pca.write(pcafile);
	
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


		for (int i =0; i < trainlist.size(); ++i)
		{
#else // linux
		for (int i = 0; i < trainlist.size(); ++i)
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

				Mat img = imread(PATH_OF_IMAGE + trainlist[i], 0);
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
	VlKMeans* getKmeansModel(int cluster_num, int feature_dim,string path_of_siftfeature,string kmeansF)
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


		//string kmeansF = "vlad_kmeans.model";
		ifstream inputF(path_of_siftfeature);
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
		vl_size dimension = -1;
		vl_size numClusters = -1;
		VlKMeans *kmeans = vlad::initVladEngine("vlad128_sift32.model", dimension, numClusters);
		//VlKMeans *kmeans = new VlKMeans();
		//VlKMeans*kmeans = getKmeansModel(NUMBER_OF_CLUSTERS, FEA_DIM, "vlsift_tmp.fea");
		//vl_size dimension = vl_kmeans_get_dimension(kmeans);
		//vl_size numClusters = vl_kmeans_get_num_centers(kmeans);
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
		for (int i = 0; i < trainlist.size();++i)//^M
		{//^M
#else // linux
		for (int i = 0; i < trainlist.size(); ++i)
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
				Mat imgS = imread(PATH_OF_IMAGE + trainlist[i], 0);
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
	void  GetVladFeatureFromSift(string kmeansfile,string testlistfile)
	{
		String aa = "D:/E/work/dressplus/code/AlgorithmTestSystem/AlgorithmTestSystem/data/featurelist.txt";
		ifstream inputF(aa);
		if (inputF.is_open())
		{
			cout<<"can not open the list"<<endl;
		}
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
		vl_size dimension = -1;
		vl_size numClusters = -1;
		VlKMeans *kmeans = vlad::initVladEngine(PATH_OF_SIFTFEATURE+kmeansfile, dimension, numClusters);
		//VlKMeans *kmeans = new VlKMeans();
		//VlKMeans*kmeans = getKmeansModel(NUMBER_OF_CLUSTERS, FEA_DIM);
		//vl_size dimension = vl_kmeans_get_dimension(kmeans);
		//vl_size numClusters = vl_kmeans_get_num_centers(kmeans);
		//------
		string imagename;
		int cnt = 0;
		int img_number = 0;
		vector<string> trainlist;
		string ss;
		getline(inputF,ss);
		while (getline(inputF, imagename))
			trainlist.push_back(imagename);
#ifdef linux
		//#pragma omp parallel for
		for (int i = 0; i < trainlist.size(); ++i)//^M
		{//^M
#else // linux
//#pragma omp parallel for
		for (int i = 0; i < trainlist.size(); ++i)
		{
#endif
			if (cnt++ % 100 == 0)
				cout << "proc " << cnt << endl;
			if (img_number > NUMBER_OF_IMAGES_TO_TEST)
			{
				//break;
			}
		//	try{
				img_number++;
				string sift_path = PATH_OF_SIFTFEATURE + trainlist[i];
				ifstream sift_image(sift_path.c_str());
				vector<float> sift_feature;
				string single_sift_feature;
				string info_of_image;
				getline(sift_image, info_of_image);
				cout << "image " << trainlist[i] << " " << info_of_image << endl;
				int tmp_line = 2;
				while (getline(sift_image,single_sift_feature))
				{
					vector<float> sift_feature_tmp;
					/*vector<string> sift_elements;
					split_words(single_sift_feature," ",sift_elements);
					for (int ii = 2; ii < sift_elements.size();++ii)
					{
					sift_feature.push_back(atof(sift_elements[i].c_str()));
					}*/
					istringstream single_sift_stream(single_sift_feature);
					string x, y;
					single_sift_stream >> x;
					single_sift_stream >> y;
				//	cout << x << "   " << y;
					float tempnumber;
					int temp_count = 0;
					//while (single_sift_stream>>tempnumber)
					//{
					//	temp_count++;
					//	sift_feature_tmp.push_back(tempnumber);
					////	cout << tempnumber << endl;
					//}
					for (int i = 0; i < 128; ++i)
					{
						temp_count++;
						single_sift_stream >> tempnumber;
						sift_feature_tmp.push_back(tempnumber);
					}
					if (temp_count!=FEA_DIM)
					{
						cout << "this line can not be parse righrly " << tmp_line << endl;
						cout << temp_count << endl;
						continue;
					}
					if (tmp_line == 20)
					{
						cout << "20th line" << endl;
						for each (float var in sift_feature_tmp)
						{
							cout << var << " ";
						}
						cout << endl;
					}
					else
					{
						for (int i = 0; i < FEA_DIM;++i)
						{
							sift_feature.push_back(sift_feature_tmp.at(i));
						}
					}
					//cout << tmp_line << endl;
					//cout << tempnumber<<endl;
					tmp_line++;
					
					

				}
				cout << sift_feature.size() << endl;
				cout <<sift_feature.at(0)<<" "<< sift_feature.at(sift_feature.size() - 1) << endl;
				int len = sift_feature.size() / dimension;
				assert(sift_feature.size() % dimension == 0);
				vector<float> vlf = vlad::encodeVladFea(kmeans, sift_feature, dimension, numClusters);
				assert(dimension*numClusters == vlf.size());
				vector<string> inputfile;
				split_words(trainlist[i], ".", inputfile);

				string outputname = PATH_OF_VLADFEATURE+inputfile[0]+".vladfea";
				ofstream vladsavefile(outputname.c_str());
				for (int i = 0; i < vlf.size();++i)
				{
					vladsavefile << vlf.at(i);
				}
				vladsavefile << endl;
				vladsavefile.close();
				sift_image.close();
		//	}
		//	catch (...){
		//		cout << "there are something wrong with picture" << PATH_OF_IMAGE << line << endl;
		//		continue;
		//	}
		
		}
			vl_kmeans_delete(kmeans);
			inputF.close();
		}
	}