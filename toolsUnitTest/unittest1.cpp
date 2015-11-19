#include "stdafx.h"
#include "CppUnitTest.h"
#include "utls.h"
#include "encode2binary.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace toolsUnitTest
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(SPLIT_WORDS)
		{
			vector<double>true_result{ 0.162182, 0.794285, 0.311215, 0.528533, 0.165649, 0.601982, 0.262971, 0.654079, 0.689215, 0.748152 };
			vector<string>myresult;
			ifstream datatest("a.txt");
			if (!datatest)
			{
				cout << "data can not be read" << endl;
			}
			string line = "0.162182 0.794285 0.311215 0.528533 0.165649 0.601982 0.262971 0.654079 0.689215 0.748152";
			split_words(line," ",myresult);
			int counter = 0;
			for (int i = 0; i < myresult.size();++i)
			{
				if (true_result[i] == atof(myresult[i].c_str()))
				{
					counter++;
				}
			}
			Assert::AreEqual(10, counter);

		}
		TEST_METHOD(BINARYENCODE)
		{
			Mat model = Mat(3, 2, CV_32FC1);
			float model_temp[3][2]{{-1.0, 1.0}, { 1.0, -1.0 }, { 1.0, -1.0 }};
		
		for (int i = 0; i < 3;++i)
			{
				for (int j = 0; j < 2;++j)
				{
					model.at<float>(i, j) = model_temp[i][j];
				}
			}
		Mat rawdata = Mat(1, 3, CV_32FC1, Scalar::all(1));
		Mat result1,result2;
		encode2Binary(rawdata,model, result1);
		vector<float> rawdata2{ 1.0, 1.0, 1.0 };
		encode2Binary(rawdata2,model,result2);
		if (result1.at<float>(0,0)==1&&result1.at<float>(0,1)==0)
		{
			if (result2.at<float>(0, 0) == 1 && result2.at<float>(0, 1) == 0)
			{
				Assert::AreEqual(1, 1);
			}
			else
			{
				Assert::AreEqual(1, 0);
			}
		}
		else
		{
			Assert::AreEqual(1, 0);
		}
		}

	};
}