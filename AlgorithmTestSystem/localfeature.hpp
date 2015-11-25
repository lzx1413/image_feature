#ifndef LOCAL_FEATURE_HPP
#define LOCAL_FEATURE_HPP
#include <opencv2/opencv.hpp>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/dsift.h>
#include <vl/sift.h>
#include <vl/vlad.h>

#include "utls.h"
vector<float> genDescriptorReduced(Mat& descriptors, Mat& mlModel)
{
	Mat dmat;
	L2NormFeature(descriptors);
	do_metric(mlModel, descriptors, dmat);
	return getVector(dmat);
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
	img.assignTo(img, CV_32FC1);///转换数据类型
	//float *im = (float*)malloc(img.cols*img.rows*sizeof(float));兼容vlfeat的接口进行数据转换
	//for (int i = 0; i < img.rows; i++)
	//	for (int j = 0; j < img.cols; j++)
	//		im[i*img.cols + j] = float(img.at<uchar>(i, j));
	float *im = (float*)img.ptr();///直接引用数据，指针无需释放
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

	//free(im);//只是引用的图的数据不需要自己管理
}
void extSparseVlSiftDes(Mat &sImg, Mat& descriptors)
{
	float *descriptor = nullptr;
	float *keyPos = nullptr;
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
	SiftFilt = nullptr;
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

	//转换类型
	Image.assignTo(Image, CV_32FC1);
	ImageData = (float*)(Image.ptr());///usding this funtion to replace the next one code block
	/*for (i = 0; i < Image.rows; i++)
	{
	for (j = 0; j < Image.cols; j++)
	ImageData[i*Image.cols + j] = float(Image.at<uchar>(i, j));
	}
	*/
	vector<float> local_contrast;
	//初始化特征
	SiftFilt = vl_sift_new(Image.cols, Image.rows, 4, 3, 0);

	if (vl_sift_process_first_octave(SiftFilt, ImageData) != VL_ERR_EOF)
	{
		while (1)
		{
			//计算每组中的关键点
			vl_sift_detect(SiftFilt);
			//遍历并绘制每个点
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
			//下一阶
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
	descriptor = nullptr;
	vl_sift_delete(SiftFilt);
}
#endif // !LOCAL_FEATURE_HPP
