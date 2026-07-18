#pragma once
#include <string>
#include <vector>

using namespace std;

struct AudioBuffer {
    vector<float> samples;
    int sample_rate = 0;
};

bool record_audio_chunk(const string& output_path, int duration_sec);
AudioBuffer load_wav(const string& path);
