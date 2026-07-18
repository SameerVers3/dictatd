#include "audio.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

using namespace std;

bool record_audio_chunk(const string& output_path, int duration_sec) {
    ostringstream cmd;
    cmd << "arecord -f S16_LE -r 16000 -c 1 -d " << duration_sec
        << " \"" << output_path << "\" 2>/dev/null";
    Logger::log("RECORD", "Executing: " + cmd.str());
    if (system(cmd.str().c_str()) == 0) {
        return true;
    }
    Logger::log("RECORD", "arecord failed, trying ffmpeg...");
    ostringstream cmd2;
    cmd2 << "ffmpeg -f alsa -i default -ar 16000 -ac 1 -t " << duration_sec
         << " \"" << output_path << "\" -y 2>/dev/null";
    Logger::log("RECORD", "Executing: " + cmd2.str());
    return system(cmd2.str().c_str()) == 0;
}

AudioBuffer load_wav(const string& path) {
    AudioBuffer audio;
    ifstream file(path, ios::binary);
    if (!file) {
        Logger::log("WAV", "ERROR: Cannot open file: " + path);
        return audio;
    }
    auto read_u32 = [&file]() -> uint32_t {
        uint8_t buf[4];
        file.read(reinterpret_cast<char*>(buf), 4);
        return buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
    };
    auto read_u16 = [&file]() -> uint16_t {
        uint8_t buf[2];
        file.read(reinterpret_cast<char*>(buf), 2);
        return buf[0] | (buf[1] << 8);
    };
    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        Logger::log("WAV", "ERROR: Not a RIFF file");
        return audio;
    }
    read_u32();
    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        Logger::log("WAV", "ERROR: Not a WAVE file");
        return audio;
    }
    while (file) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        if (!file) break;
        uint32_t chunk_size = read_u32();
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = read_u16();
            uint16_t num_channels = read_u16();
            uint32_t sample_rate = read_u32();
            read_u32();
            read_u16();
            uint16_t bits_per_sample = read_u16();
            audio.sample_rate = static_cast<int>(sample_rate);
            Logger::log("WAV", "Format: " + to_string(audio_format)
                      + ", Channels: " + to_string(num_channels)
                      + ", Rate: " + to_string(sample_rate)
                      + ", Bits: " + to_string(bits_per_sample));
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, ios::cur);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            size_t num_samples = chunk_size / 2;
            vector<int16_t> pcm(num_samples);
            file.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            audio.samples.reserve(num_samples);
            for (int16_t s : pcm) {
                audio.samples.push_back(s / 32768.0f);
            }
            Logger::log("WAV", "Loaded " + to_string(num_samples)
                      + " samples from " + path);
            break;
        } else {
            file.seekg(chunk_size, ios::cur);
        }
    }
    return audio;
}
