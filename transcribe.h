#pragma once
#include <string>

using namespace std;

string transcribe_with_whisper(const string& audio_path, const string& model_path);
