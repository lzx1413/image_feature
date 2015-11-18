#include "stdafx.h"
#include "CppUnitTest.h"
#include"utls.h"
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

	};
}