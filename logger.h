#pragma once
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace std;

class Logger {
public:
    static void log(const string& stage, const string& msg);
};
