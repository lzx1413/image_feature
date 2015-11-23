#pragma once
#include "encode2Bianry.hpp"
class MethodTimeResume
{
public:
	explicit MethodTimeResume(string timelogfile);
	MethodTimeResume(const MethodTimeResume&) = delete;
	MethodTimeResume& operator =(const MethodTimeResume&) = delete;
	~MethodTimeResume();
    void testMatrixMulti();
	void test();
private:
	string time_log_file;
	ofstream time_log;
	void testEncode2Binary();
};

