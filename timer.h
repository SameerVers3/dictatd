#pragma once
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace std;

class Timer {
    chrono::high_resolution_clock::time_point start_;
    string name_;
public:
    explicit Timer(const string& name);
    ~Timer();
    double elapsed_sec() const;
    static string format_time(double secs);
};
