
#ifndef _LOG_INCLUDE_H_
#define _LOG_INCLUDE_H_

#define LOG_LINE_MAX 1024   

#include <fstream>

using std::string;
using std::fstream;
using std::ios;

class CLog
{
public:
	enum LOG_LEVEL    
	{
		LL_ERROR = 1,              
		LL_WARN = 2,               
		LL_INFORMATION = 3         
	};

public:
    CLog(void):m_openSuccess(false), m_LogLevel(LL_ERROR), m_showLogFlag(true), 
		m_maxLogFileSize(10 * 1024 *1024)
    {
    }
    CLog(const char *filePath, LOG_LEVEL level = LL_ERROR);
	CLog(const wchar_t *filePath, LOG_LEVEL level = LL_ERROR);
    virtual ~CLog(void)
    {
        if (m_openSuccess)
        {
			CloseLogFile();
        }
    }

    bool GetOpenStatus() const
    {
        return m_openSuccess;
    }


    void OpenLogFile(const char *pFilePath, LOG_LEVEL level = LL_ERROR);
    void OpenLogFile(const wchar_t *pFilePath, LOG_LEVEL level = LL_ERROR);

    void WriteLog(LOG_LEVEL level, const char *pLogText, ...);
    void WriteLog(string logText, LOG_LEVEL level = LL_ERROR);
    void WriteLogEx(LOG_LEVEL level, const char *pLogText, ...);

	size_t GetLogFileSize();
	
	void ClearLogFile();
	void CloseLogFile();

    void SetLogLevel(LOG_LEVEL level = LL_ERROR)
    {
        m_LogLevel = level;
    }
    LOG_LEVEL GetLogLevel() const
    {
        return m_LogLevel;
    }

    void SetShowFlag(bool flag = true)
    {
        m_showLogFlag = flag;
    }
    bool GetShowFlag() const
    {
        return m_showLogFlag;
    }

    void SetMaxLogFileSize(size_t size)
    {
        m_maxLogFileSize = size;
    }
    size_t GetMaxLogFileSize() const
    {
        return m_maxLogFileSize;
    }

private:
    CLog(const CLog &clog){};

protected:
	string W2A(const wchar_t *pWcsStr);  
    string ConvertToRealLogText(const char *pLogText, LOG_LEVEL level = LL_ERROR);
    const std::string &StringFormat(std::string &srcString, const char *pFormatString, ...);
protected:
    fstream m_fileOut;
    bool m_openSuccess;  
	string m_logFilePath; 

protected:
    LOG_LEVEL m_LogLevel;  
	bool m_showLogFlag;    
	size_t m_maxLogFileSize; 
};

#endif

