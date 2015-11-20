#pragma once
#include "encode2binary.h"
class MethodTimeResume
{
public:
	explicit MethodTimeResume(string timelogfile);
	MethodTimeResume(const MethodTimeResume&) = delete;
	MethodTimeResume& operator =(const MethodTimeResume&) = delete;
	~MethodTimeResume();
	void test();
private:
	string time_log_file;
	ofstream time_log;
	void testEncode2Binary();
};

