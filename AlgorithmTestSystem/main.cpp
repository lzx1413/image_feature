// TestVLAD_FV.cpp : �������̨Ӧ�ó������ڵ㡣
//
//include the files form opencv
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
//stl
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <assert.h>
//Include the files from the vl
#include <vlad.h>
#include <fisher.h>
#include <kmeans.h>
#include <gmm.h>
#include <dsift.h>
#include <sift.h>
#include <vlad.h>

using namespace std;
using namespace cv;
const static int NUMBER_OF_IMAGES = 100;
const static string PATH_OF_WORK = "D:/E/work/dressplus/code/temp/";

void RootNormFeature(vector<float>& sdes);
void L2NormFeature(Mat& smat, int rowidx);
void L2NormFeature(Mat& smat);
void saveGmmModel(const char * modelFile, VlGMM * gmm);
void loadGmmModel(const char * modelFile, VlGMM *& gmm, vl_size & dimension, vl_size & numClusters);
void loadKmeansModel(const char * modelFile, float*& km, vl_size & dimension, vl_size & numClusters);
void saveKmeansModel(const char * modelFile, VlKMeans * km);
void split_words(const string& src, const string& separator, vector<string>& dest);
bool load_metric_model(string filePath, Mat& ml_model);
void do_metric(Mat& mlModel, Mat& smat, Mat& dmat);
void extDenseVlSiftDes(Mat& sImg, Mat& descriptors);
vector<float> getVector(const Mat &_t1f);
vector<float> genVlEncodeFormat(Mat& descriptors, Mat& mlModel);
void extSparseVlSiftDes(Mat &sImg, Mat& descriptors);
PCA compressPCA(const Mat& pcaset, int maxComponents,const Mat& testset, Mat& compressed);
//INIT VLAD ENGINE
VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum);
vector<float> encodeVladFea(VlKMeans *vladModel, vector<float> rawFea, int feature_dim, int clusterNum);



// patch-wise mean subtraction
Mat patchWiseSubtraction(Mat sImg)
{
	Mat dImg;
	// patch-wise mean subtraction
	cv::Scalar mean, stddev;
	cv::meanStdDev(sImg, mean, stddev);
	{
		cv::Mat *slice = new cv::Mat[sImg.channels()];
		cv::split(sImg, slice);
		for (int c = 0; c < sImg.channels(); ++c) {
			cv::subtract(slice[c], mean[c], slice[c]);
		}
		cv::merge(slice, sImg.channels(), dImg);
		delete[] slice;
	}

	return dImg;
}


// patch-wise stddev division
Mat patchWiseStdDevDiv(Mat sImg)
{
	Mat dImg;
	// patch-wise stddev division
	cv::Scalar mean, stddev;
	cv::meanStdDev(sImg, mean, stddev);

	cv::Mat *slice = new cv::Mat[sImg.channels()];
	cv::split(sImg, slice);
	for (int c = 0; c < sImg.channels(); ++c) {
		slice[c] /= stddev[c];
	}
	cv::merge(slice, dImg.channels(), dImg);
	delete[] slice;

	return dImg;
}


#include <ctime>
#define SELECT_NUM 1000
#define FEA_DIM 32
//#define USE_PCA
//#define EXTSIFT_PHASE
//#define TRAIN_PHASE
//#define TEST_PHASE
void ExitTheSiftFeature(string trainlistfile)
{
	ifstream inputtrainlistF(trainlistfile.c_str());
	// load model 
	Mat mlModel;
	load_metric_model(PATH_OF_WORK+"DimentionReduceMat_vlsift_32.txt", mlModel);

	ofstream outputF("vlsift_tmp.fea");
	string line;
	int cnt = 0;
	int img_number = 0;
	Mat descriptor_set;
	while (getline(inputtrainlistF, line))
	{
		if (cnt++ % 100 == 0)
			cout << "proc " << cnt << endl;
		if (img_number > NUMBER_OF_IMAGES)
		{
			break;
		}
		img_number++;
		srand((unsigned)getTickCount());

		Mat img = imread("D:/E/work/dressplus/code/data/fvtraindata/" + line, 0);
		if (img.empty() || img.cols < 64 || img.rows < 64)
			continue;
		double t = (double)cv::getTickCount();
		Mat descriptors;
		extDenseVlSiftDes(img, descriptors);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << t << " s" << std::endl;
#ifdef USE_PCA

		descriptor_set.push_back(descriptors);
#else
		vector<float> normfea = genVlEncodeFormat(descriptors, mlModel);
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
#endif//USE_PCA
	}
	cout << "proc " << cnt << endl;
#ifdef USE_PCA
	L2NormFeature(descriptor_set);
	Mat descriptor_set_after_pca;
	compressPCA(descriptor_set, 32, descriptor_set, descriptor_set_after_pca);
	for (auto i = 0; i < descriptor_set_after_pca.rows; ++i)
	{
		for (auto c = 0; c < descriptor_set_after_pca.cols; ++c)
		{
			auto data = descriptor_set_after_pca.at<float>(i, c);
			outputF << data;
		}
		outputF << endl;
	}

#endif // USE_PCA

	inputtrainlistF.close();
	outputF.close();
}
void TrainVladModel()
{
	// set params
	vl_set_num_threads(0); /* use the default number of threads */

	vl_size dimension = FEA_DIM;
	vl_size numClusters = 512;
	vl_size maxiter = 128;
	vl_size maxrep = 1;

	VlKMeans * kmeans = 0;
	vl_size maxiterKM = 64;
	vl_size ntrees = 3;
	vl_size maxComp = 64;

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
	saveKmeansModel(kmeansF.c_str(), kmeans);
	cout << "Save model to " << kmeansF << endl;

	if (kmeans) {
		vl_kmeans_delete(kmeans);
	}
}

int TestVladModel(string testlistfile)
{
	ifstream inputF(testlistfile.c_str());
	// load model 
	Mat mlModel;
	bool flag = load_metric_model(PATH_OF_WORK + "DimentionReduceMat_vlsift_32.txt", mlModel);
	if (!flag)
		return -1;
	// load VLAD model
	vl_size dimension = -1;
	vl_size numClusters = -1;
	VlKMeans *kmeans = initVladEngine("vlad128_sift32.model", dimension, numClusters);

	//------
	ofstream outputF( "vlad_sift32.test.fea");
	string line;
	int cnt = 0;
	int img_number = 0;
	while (getline(inputF, line))
	{
		if (cnt++ % 100 == 0)
			cout << "proc " << cnt << endl;
		if (img_number > NUMBER_OF_IMAGES)
		{
			break;
		}
		img_number++;
		Mat imgS = imread("D:/E/work/dressplus/code/data/fvtraindata/" + line, 0);
		if (imgS.empty() || imgS.cols < 64 || imgS.rows < 64)
			continue;
		// norm image
		int normWidth = 360;
		int normHeight = 360.0 / imgS.size().width*imgS.size().height;
		Mat img;
		resize(imgS, img, Size(normWidth, normHeight));

		double t = (double)cv::getTickCount();
		Mat descriptors;
		extDenseVlSiftDes(img, descriptors);
		//extSparseVlSiftDes(img, descriptors);
		vector<float> normfea = genVlEncodeFormat(descriptors, mlModel);

		int len = normfea.size() / dimension;
		assert(normfea.size() % dimension == 0);

		vector<float> vlf = encodeVladFea(kmeans, normfea, dimension, numClusters);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << t << " s" << std::endl;

		assert(dimension*numClusters == vlf.size());

		outputF << line;
		for (int i = 0; i < vlf.size(); i++)
			outputF << " " << vlf[i];
		outputF << endl;
	}

	vl_kmeans_delete(kmeans);
	inputF.close();
	outputF.close();
}
void help()
{
	cout << "Ussage:--trainlist <trianlist \n\t --testlist <testlist>" << endl;
}
int main(int argc, char* argv[])
{
	string trainlistfile;
	string testlistfile;
	if (argc == 1)
	{
		help();
		return -1;
	}
	for (int i = 1; i < argc;++i)
	{
		
		if (string(argv[i])=="--trainlist")
		{
			trainlistfile = argv[++i];
		}
		else if (string(argv[i]) == "--testlist")
		{
			testlistfile = argv[++i];
		}

	}
	ExitTheSiftFeature(trainlistfile);
	TrainVladModel();
	TestVladModel(testlistfile);
	
	cout << "work has been down" << endl;
	getchar();
	return 0;
}




void split_words(const string& src, const string& separator, vector<string>& dest)
{
	dest.clear();
	string str = src;
	string substring;
	string::size_type start = 0, index;

	do
	{
		index = str.find_first_of(separator, start);
		if (index != string::npos)
		{
			substring = str.substr(start, index - start);
			dest.push_back(substring);
			start = str.find_first_not_of(separator, index);
			if (start == string::npos) return;
		}
	} while (index != string::npos);

	//the last token
	substring = str.substr(start);
	dest.push_back(substring);
}


void saveGmmModel(const char * modelFile, VlGMM * gmm)
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


void loadGmmModel(const char * modelFile, VlGMM *& gmm, vl_size & dimension, vl_size & numClusters)
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

void saveKmeansModel(const char * modelFile, VlKMeans * km)
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


void loadKmeansModel(const char * modelFile, float*& km, vl_size & dimension, vl_size & numClusters)
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


bool load_metric_model(string filePath, cv::Mat& ml_model)
{
	ifstream inputF(filePath.c_str());
	if (!inputF.is_open())
		return false;
	string line;
	getline(inputF, line);
	vector<string> ww;
	split_words(line, " ", ww);
	int row = atoi(ww[0].c_str());
	int col = atoi(ww[1].c_str());
	cout << "MetricMat Info: " << row << "\t" << col << endl;

	cv::Mat ml_model_ = cv::Mat(row, col, CV_32F);
	for (int i = 0; i < row; i++)
	{
		getline(inputF, line);
		split_words(line, " ", ww);
		for (int j = 0; j < col; j++){
			ml_model_.at<float>(i, j) = atof(ww[j].c_str());
		}
	}

	cv::transpose(ml_model_, ml_model);
	//ml_model = ml_model_;
	return true;
}

void do_metric(Mat& mlModel, Mat& smat, Mat& dmat)
{
	//L2 Norm
	L2NormFeature(smat);
	//reduction
	cv::gemm(smat, mlModel, 1.0, 0.0, 0.0, dmat);
}

void L2NormFeature(Mat& smat, int rowidx)
{
	int des_dim = smat.cols;
	//L2 norm
	float total = 0.0;
	for (int kk = 0; kk < des_dim; kk++)
	{
		total += smat.at<float>(rowidx, kk) * smat.at<float>(rowidx, kk);
	}
	total = total < 0.00001 ? 1 : sqrtf(total);
	for (int kk = 0; kk < des_dim; kk++)
		smat.at<float>(rowidx, kk) /= total;
}

void L2NormFeature(Mat& smat)
{
	for (int i = 0; i < smat.rows; i++)
		L2NormFeature(smat, i);
}

void RootNormFeature(vector<float>& sdes)
{
	int des_dim = sdes.size();
	float total = 0.0;
	for (int kk = 0; kk < des_dim; kk++)
	{
		total += fabs(sdes[kk]);
	}
	total = total < 0.00001 ? 1 : total;
	for (int kk = 0; kk < des_dim; kk++)
		sdes[kk] = sdes[kk] < 0 ? -1 * sqrt(-1 * sdes[kk] / total) : sqrt(sdes[kk] / total);
}

void extDenseVlSiftDes(Mat& sImg, Mat& descriptors)
{
	Rect box = Rect(1, 1, sImg.size().width - 2, sImg.size().height - 2);

	Mat img;
	if (sImg.channels() == 1)
		img = sImg;
	else{
		cv::cvtColor(sImg, img, cv::COLOR_BGR2GRAY);
	}

	float *im = (float*)malloc(img.cols*img.rows*sizeof(float));
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			im[i*img.cols + j] = float(img.at<uchar>(i, j));

	//-- set dsift parameters

	//	8.0f,     //initFeatureScale: 
	int featureScale = 8.0;
	//	3,        //featureScaleLevels
	int scaleNum = 3;
	//	1.414f,   //featureScaleMul
	float scaleMul = 1.414;
	//	4,        //initXyStep2dx
	int xyStep = 4;

	//  descriptor
	int dim = -1;
	int valid_num = 0;
	vector<float> desV;
	for (int i = 0; i < scaleNum; i++)
	{
		int featureScale_ = int(featureScale*powf(scaleMul, i));
		int xyStep_ = int(xyStep*powf(scaleMul, i));
		VlDsiftFilter *filter = vl_dsift_new_basic(img.cols, img.rows, xyStep_, featureScale_);
		vl_dsift_set_bounds(filter, box.x, box.y, box.x + box.width, box.y + box.height);
		vl_dsift_set_flat_window(filter, true);  //flat_window is faster than gaussian smooth

		// run core
		vl_dsift_process(filter, im);
		// get descriptor
		VlDsiftKeypoint const *he = vl_dsift_get_keypoints(filter);
		dim = vl_dsift_get_descriptor_size(filter);
		int num = vl_dsift_get_keypoint_num(filter);
		float const * des = vl_dsift_get_descriptors(filter);

		for (int k = 0; k < num; k++){
			//copy descriptor
			for (int kk = 0; kk < dim; kk++){
				desV.push_back(des[kk]);
			}
			des += dim;
			valid_num += 1;
		}

		vl_dsift_delete(filter);
	}
	descriptors = Mat(desV, true).reshape(0, valid_num);

	free(im);
}

vector<float> getVector(const Mat &_t1f)
{
	Mat t1f;
	_t1f.convertTo(t1f, CV_32F);
	return (vector<float>)(t1f.reshape(1, 1));
}

vector<float> genVlEncodeFormat(Mat& descriptors, Mat& mlModel)
{
	Mat dmat;
	do_metric(mlModel, descriptors, dmat);
	return getVector(dmat);
}


VlKMeans * initVladEngine(string vlad_km_path, vl_size& feature_dim, vl_size& clusterNum)
{
	// load VLAD model
	vl_size dimension = -1;
	vl_size numClusters = -1;
	float* vlad_km = NULL;
	loadKmeansModel(vlad_km_path.c_str(), vlad_km, dimension, numClusters);
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
	// find nearest cluster centers for the data that should be encoded
	vl_uint32* idx_v = (vl_uint32*)vl_malloc(sizeof(vl_uint32) * elemSize);
	float* dists = (float*)vl_malloc(sizeof(float) * elemSize);
	vl_kmeans_quantize(vladModel, idx_v, dists, rawFea.data(), elemSize);
	// convert indexes array to assignments array,
	// which can be processed by vl_vlad_encode
	float* assignments = (float*)vl_malloc(sizeof(float) * elemSize * clusterNum);
	memset(assignments, 0, sizeof(float) * elemSize * clusterNum);
	for (int i = 0; i < elemSize; i++) {
		assignments[i * clusterNum + idx_v[i]] = 1.;
	}
	// do the encoding job
	vl_vlad_encode(enc, VL_TYPE_FLOAT,
		vl_kmeans_get_centers(vladModel), feature_dim, clusterNum,
		rawFea.data(), elemSize,
		assignments,
		0);

	vector<float> vladFea(feature_dim*clusterNum, 0);
	memcpy(vladFea.data(), enc, feature_dim*clusterNum*sizeof(float));

	vl_free(dists);
	vl_free(assignments);
	vl_free(idx_v);
	vl_free(enc);

	return vladFea;
}


void extSparseVlSiftDes(Mat &sImg, Mat& descriptors)
{
	float *descriptor = NULL;
	float *keyPos = NULL;
	int ipNum = 0;

	cv::Mat Image;
	if (sImg.channels() == 1)
		Image = sImg;
	else{
		cv::cvtColor(sImg, Image, cv::COLOR_BGR2GRAY);
	}

	int FEATUREDIM = 128;
	vl_sift_pix *ImageData;
	unsigned char *Pixel;
	int KeyPoint, LKeyPoint, k, i, j;
	double angles[4];
	VlSiftFilt *SiftFilt;
	float *tdescriptor, *top;

	//initializing
	SiftFilt = NULL;
	ImageData = new vl_sift_pix[Image.rows*Image.cols];

	ipNum = 0;
	KeyPoint = 0;
	LKeyPoint = 0;

	//Eliminating low-contrast descriptors. 
	//Near-uniform patches do not yield stable keypoints or descriptors. 
	//vl_sift_set_norm_thresh() can be used to set a threshold on the 
	//average norm of the local gradient to zero-out descriptors that 
	//correspond to very low contrast regions. By default, the threshold 
	//is equal to zero, which means that no descriptor is zeroed. Normally 
	//this option is useful only with custom keypoints, as detected keypoints 
	//are implicitly selected at high contrast image regions.

	//ת������
	for (i = 0; i < Image.rows; i++)
	{
		for (j = 0; j < Image.cols; j++)
			ImageData[i*Image.cols + j] = float(Image.at<uchar>(i, j));
	}

	vector<float> local_contrast;
	//��ʼ������
	SiftFilt = vl_sift_new(Image.cols, Image.rows, 4, 3, 0);

	if (vl_sift_process_first_octave(SiftFilt, ImageData) != VL_ERR_EOF)
	{
		while (1)
		{
			//����ÿ���еĹؼ���
			vl_sift_detect(SiftFilt);
			//����������ÿ����
			LKeyPoint = KeyPoint;
			KeyPoint += SiftFilt->nkeys;

			if (LKeyPoint == 0)
			{
				if (KeyPoint != 0){
					descriptor = new float[KeyPoint*FEATUREDIM];
					keyPos = new float[KeyPoint * 2];
				}
			}
			else{
				tdescriptor = new float[LKeyPoint*FEATUREDIM];
				memcpy(tdescriptor, descriptor, sizeof(float)*FEATUREDIM*LKeyPoint);
				delete[]descriptor;
				descriptor = new float[KeyPoint*FEATUREDIM];
				memcpy(descriptor, tdescriptor, sizeof(float)*FEATUREDIM*LKeyPoint);
				delete[]tdescriptor;
				top = new float[LKeyPoint * 2];
				memcpy(top, keyPos, sizeof(float) * 2 * LKeyPoint);
				delete[]keyPos;
				keyPos = new float[KeyPoint * 2];
				memcpy(keyPos, top, sizeof(float) * 2 * LKeyPoint);
				delete[]top;
			}

			VlSiftKeypoint *pKeyPoint = SiftFilt->keys;

			for (int i = 0; i < SiftFilt->nkeys; i++)
			{
				VlSiftKeypoint TemptKeyPoint = *pKeyPoint;
				pKeyPoint++;

				vl_sift_calc_keypoint_orientations(SiftFilt, angles, &TemptKeyPoint);
				float *Descriptors = new float[FEATUREDIM];
				double TemptAngle = angles[0];
				vl_sift_calc_keypoint_descriptor(SiftFilt, Descriptors, &TemptKeyPoint, TemptAngle);

				*(keyPos + (LKeyPoint + i) * 2) = TemptKeyPoint.x;
				*(keyPos + (LKeyPoint + i) * 2 + 1) = TemptKeyPoint.y;

				memcpy(descriptor + (LKeyPoint + i)*FEATUREDIM, Descriptors, FEATUREDIM*sizeof(float));
				ipNum++;
				delete[]Descriptors;
				Descriptors = NULL;

			}
			//��һ��
			if (vl_sift_process_next_octave(SiftFilt) == VL_ERR_EOF)
			{
				break;
			}
		}
	}

	descriptors = cv::Mat::zeros(ipNum, FEATUREDIM, CV_32F);
	for (int i = 0; i < ipNum; i++)
	{
		for (int ii = 0; ii < FEATUREDIM; ii++){
			descriptors.at<float>(i, ii) = descriptor[i*FEATUREDIM + ii];
		}
	}

	delete[] descriptor;
	descriptor = NULL;
	vl_sift_delete(SiftFilt);
	delete[] ImageData;
	ImageData = NULL;

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
	FileStorage pcafile(PATH_OF_WORK + "pca_model.yml", FileStorage::WRITE);
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