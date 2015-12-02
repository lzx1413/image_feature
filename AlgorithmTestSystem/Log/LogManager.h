
#ifndef _LOGMANAGER_INCLUDE_H_
#define _LOGMANAGER_INCLUDE_H_

#include <map>
#include <string>

class CLog;

class LogManager
{
public:
    static CLog* OpenLog(const char *pFilePath, int nLogLevel = 1); 
    static void Clear();                                           
    static void RemoveLog(const std::string &strLogFilePath);      

private:  
    LogManager(void);
    LogManager(const LogManager &logManager);

private:  
    static std::map<std::string, CLog*> m_logMap; 

};

#endif
