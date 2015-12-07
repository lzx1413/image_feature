#include "vladmethods.h"
#include <iostream>
#include <fstream>
#include "Config.h"
#include "cluteranalysis.hpp"
#include "localfeature.hpp"
#include "omp.h"
#include "Log/Log.h"
#include "Log/LogManager.h"
//TODO: 省略中间数据的存储和读取实现直接的使用计算
using namespace std;
const static int NUMBER_OF_IMAGES_TO_TRAIN = 10;
const static int NUMBER_OF_IMAGES_TO_TEST = 100;
static string PATH_OF_WORK = "D:/E/work/dressplus/code/temp/";
static string PATH_OF_IMAGE = "D:/E/work/dressplus/code/data/fvtraindata/";
static string NAME_OF_FEATUREFILE = "vladfeature.txt";
static string PATH_OF_SIFTFEATURE = "D:/E/work/dressplus/code/AlgorithmTestSystem/AlgorithmTestSystem/data/";
static string PATH_OF_VLADFEATURE = "";
static string PATH_OF_PCAMODEL = "";
static CLog *mylog = nullptr;
const static int SELECT_NUM = 1000;
static int  FEA_DIM = 128;
static int NUMBER_OF_CLUSTERS = 256;
const static int NUMBER_OF_IMAGES_TO_PCA = 100;

//#define REDUCE_FEATURE
//#define USE_PCA
//#define linux
#define white
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
		PATH_OF_SIFTFEATURE = configSettings.Read("path_of_siftfeature", PATH_OF_SIFTFEATURE);
		PATH_OF_VLADFEATURE = configSettings.Read("path_of_vladfeature", PATH_OF_VLADFEATURE);
		PATH_OF_PCAMODEL = configSettings.Read("path_of_pcamodel", PATH_OF_PCAMODEL);
		string path_of_log = configSettings.Read("path_of_log", string(" "));
		mylog = LogManager::OpenLog(path_of_log.c_str());
		mylog->WriteLog("configuration of the vlad completed");
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
		mylog->WriteLog("sucessfully init the vlad engine", CLog::LL_INFORMATION);
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


    //************************************
	// Method:    getPCAmodelFromSIFT
	// FullName:  vlad::getPCAmodelFromSIFT
	// Access:    public 
	// Returns:   cv::PCA
	// Qualifier:
	// Parameter: string trainlistfile trainlistfile the list of the sift feature 
	// Parameter: int maxComponents  the dimension of the pca
	// Parameter: string resultpath
	//************************************
	PCA getPCAmodelFromSIFT(string trainlistfile, int maxComponents, string resultpath)
	{
		//string kmeansF = "vlad_kmeans.model";
		ifstream inputF(trainlistfile);
		string line;
		vector<float> data_des;
		Mat feature_set;
		int count = 0;
		while (getline(inputF, line))
		{
	
			if (count%10==0)
			{
				try{
					vector<string> ww;
					split_words(line, " ", ww);
					assert(ww.size() == FEA_DIM);
					for (int i = 0; i < ww.size(); i++)
						//TODO: bad alloc
						data_des.push_back(atof(ww[i].c_str()));
					Mat single_fea = Mat(1, data_des.size(), CV_32FC1, data_des.data());
					feature_set.push_back(single_fea);
					data_des.clear();
				}
				catch (...){
					mylog->WriteLog("there some thing wrong of the sift data in line " + std::to_string(count), CLog::LL_ERROR);
				}
			}
		
			count++;
		}
		PCA pca(feature_set, Mat(), PCA::DATA_AS_ROW, maxComponents);
		FileStorage pcafile(resultpath, FileStorage::WRITE);
		pcafile << "mean" << pca.mean;
		pcafile << "e_vectors" << pca.eigenvectors;
		pcafile << "e_values" << pca.eigenvalues;
		mylog->WriteLog("sucessfully get the pca model from sift features saved in " + resultpath, CLog::LL_INFORMATION);
		return pca;

	}

	//************************************
	// Method:    getPCAmodelFromImage
	// FullName:  vlad::getPCAmodelFromImage
	// Access:    public 
	// Returns:   cv::PCA
	// Qualifier:
	// Parameter: string trainlistfile the list of the images
	// Parameter: int maxComponents the dimension of the pca 
	// Parameter: string result_path
	//************************************
	PCA getPCAmodelFromImage(string trainlistfile, int maxComponents = FEA_DIM,string result_path = PATH_OF_PCAMODEL)
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
				mylog->WriteLog("there are something wrong with picture " + PATH_OF_IMAGE + line);
				continue;
			}
		}
		PCA pca(descriptor_set, Mat(), PCA::DATA_AS_ROW, maxComponents);
		FileStorage pcafile(result_path, FileStorage::WRITE);
		pcafile << "mean" << pca.mean;
		pcafile << "e_vectors" << pca.eigenvectors;
		pcafile << "e_values" << pca.eigenvalues;
		mylog->WriteLog("sucessfully get the pca model from pictures saved in " + result_path, CLog::LL_INFORMATION);
		return pca;
	}


	//************************************
	// Method:    loadPCAmodel
	// FullName:  vlad::loadPCAmodel
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: const string pcafilepath
	// Parameter: PCA & pca
	//************************************
	void  loadPCAmodel(const string pcafilepath, PCA& pca)
	{
		FileStorage pcafile(pcafilepath, FileStorage::READ);
		if (!pcafile.isOpened())
		{
			std::cout << "can not open the pca model file" << endl;
			mylog->WriteLog("can not open the pca model file");
			return ;
		}
		pcafile["mean"] >> pca.mean;
		pcafile["e_vectors"] >> pca.eigenvectors;
		pcafile["e_values"] >> pca.eigenvalues;
		mylog->WriteLog("sucessfully load the pca model from " + pcafilepath, CLog::LL_INFORMATION);

	}


	//************************************
	// Method:    PCA_project_with_white
	// FullName:  vlad::PCA_project_with_white
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: InputArray & rawdata
	// Parameter: vector<float> & reduced_data
	// Parameter: PCA & pca
	//************************************
	void PCA_project_with_white(InputArray& rawdata, vector<float>& reduced_data,PCA& pca)
	{
		pca.project(rawdata, reduced_data);
		for (int i = 0; i < reduced_data.size(); ++i)
		{
			reduced_data.at(i) = reduced_data.at(i) / sqrt(pca.eigenvalues.at<float>(i, 0)+0.001);
		}
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

	//************************************
	// Method:    ExitTheSiftFeature
	// FullName:  vlad::ExitTheSiftFeature
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: string trainlistfile
	//************************************
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
				//LocalFeature::extDenseVlSiftDes(img, descriptors);
				LocalFeature::extDenseSURFDes(img, descriptors);
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
				cout << "there are something wrong with picture" << PATH_OF_IMAGE << trainlistfile.at(i) << endl;
				mylog->WriteLog("there are something wrong with picture" +PATH_OF_IMAGE+trainlistfile.at(i));
				continue;
			}
		}
		cout << "proc " << cnt << endl;
		inputtrainlistF.close();
		outputF.close();
		}




	//************************************
	// Method:    getKmeansModel
	// FullName:  vlad::getKmeansModel
	// Access:    public 
	// Returns:   VlKMeans*
	// Qualifier:
	// Parameter: int cluster_num
	// Parameter: int feature_dim
	// Parameter: string path_of_siftfeature
	// Parameter: string kmeansF
	//************************************
	VlKMeans* getKmeansModel(int cluster_num, int feature_dim,string path_of_siftfeature,string kmeansF)
	{
#ifdef USE_PCA
		PCA pca;	
		loadPCAmodel(PATH_OF_PCAMODEL, pca);
#endif
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
		vector<float> single_data;
		vector<float> reduced_data;
		while (getline(inputF, line))
		{
			vector<string> ww;
			split_words(line, " ", ww);
			assert(ww.size() == FEA_DIM);
			for (int i = 0; i < ww.size(); i++)
			{	//TODO: bad alloc
				single_data.push_back(atof(ww[i].c_str()));
			}
#ifdef USE_PCA
#ifdef white
			PCA_project_with_white(single_data, reduced_data, pca);
#else
			pca.project(single_data, reduced_data);
#endif

#endif //

			for (int i = 0; i < FEA_DIM; i++)
			{
				data_des.push_back(reduced_data.at(i));
			}
			single_data.clear();
			reduced_data.clear();
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



	//************************************
	// Method:    GetVladFeatureFromSift
	// FullName:  vlad::GetVladFeatureFromSift
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: string kmeansfile
	// Parameter: string testlistfile
	//************************************
	void  GetVladFeatureFromSift(string kmeansfile,string testlistfile)
	{
		
		ifstream inputF(testlistfile);
		if (inputF.is_open())
		{
			cout<<"can not open the list"<<endl;
			mylog->WriteLog("can not open the sift list"+testlistfile);
			return;
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

		//#pragma omp parallel for

		for (int i = 0; i < trainlist.size(); ++i)
		{

		    	if (cnt++ % 100 == 0)
			    	cout << "proc " << cnt << endl;
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
					istringstream single_sift_stream(single_sift_feature);
					string x, y;
					single_sift_stream >> x;
					single_sift_stream >> y;
					float tempnumber;
					int temp_count = 0;
					while (single_sift_stream>>tempnumber)
					{
						temp_count++;
						sift_feature_tmp.push_back(tempnumber);
					}
					if (temp_count!=FEA_DIM)
					{
						cout << "this line can not be parse righrly " << tmp_line << endl;
						mylog->WriteLog(trainlist[i] + std::to_string(tmp_line) + "can not be parsed in the right way");
						cout << temp_count << endl;
						continue;
					}
					else
					{
						for (int i = 0; i < FEA_DIM;++i)
						{
							sift_feature.push_back(sift_feature_tmp.at(i));
						}
					}
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

		}
			vl_kmeans_delete(kmeans);
			inputF.close();
		}
	}