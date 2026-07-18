#include "timer.h"
#include "logger.h"

Timer::Timer(const string& name) : name_(name) {
    start_ = chrono::high_resolution_clock::now();
    Logger::log(name_, "STARTED");
}

Timer::~Timer() {
    auto end = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(end - start_).count();
    Logger::log(name_, "COMPLETED in " + format_time(secs));
}

double Timer::elapsed_sec() const {
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start_).count();
}

string Timer::format_time(double secs) {
    ostringstream oss;
    if (secs < 1.0) {
        oss << fixed << setprecision(1) << (secs * 1000.0) << " ms";
    } else {
        oss << fixed << setprecision(2) << secs << " s";
    }
    return oss.str();
}
