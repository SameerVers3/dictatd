#include "logger.h"

void Logger::log(const string& stage, const string& msg) {
    
    auto now = chrono::system_clock::now();
    
    auto time_t_now = chrono::system_clock::to_time_t(now);
    
    tm tm_info{};
    localtime_r(&time_t_now, &tm_info);
    
    char time_buf[32];
    
    strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_info);
    
    cout << "[" << time_buf << "] [" << left << setw(14)
         << stage << "] " << msg << endl;
}
