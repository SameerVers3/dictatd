// dictatd: self-contained launcher/daemon manager for the voice pipeline.
//
//   - ensures ~/.dictatd/{models,config,log,pid} exist
//   - downloads missing whisper/llama models on demand (resumable)
//   - starts voice_pipeline as a background daemon, or stops/restarts it
//
// Commands:
//   setup           create dirs + config and download any missing models
//   start           setup (if needed) then launch the daemon
//   stop            stop the daemon
//   restart         stop + start
//   status          show model state and whether the daemon is running
//   log             tail the daemon log
//   systemd         write a user systemd unit for the daemon
//   (no command)    setup then start
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>
#include <sstream>

using namespace std;

static const char* kWhisperUrl =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
static const char* kLlamaUrl =
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf";

// Minimum acceptable sizes (bytes) to consider a model "installed".
static const long kWhisperMin = 100000000; // 141 MB total
static const long kLlamaMin   = 300000000; // 396 MB total

struct Paths {
    string home;
    string root;        // ~/.dictatd
    string models;      // ~/.dictatd/models
    string config;      // ~/.dictatd/config
    string log;         // ~/.dictatd/dictatd.log
    string pid;         // ~/.dictatd/dictatd.pid
    string self;        // dir containing this binary (and voice_pipeline)
    string pipeline;    // path to the voice_pipeline core
};

struct Model {
    string key;         // config key
    string file;        // full path on disk
    string url;
    string display;     // short name
    long min_bytes;
};

static string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static bool file_size(const string& path, long* size) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    *size = st.st_size;
    return S_ISREG(st.st_mode);
}

static bool is_running(pid_t pid) {
    if (pid <= 0) return false;
    return kill(pid, 0) == 0;
}

static bool mkdir_p(const string& path) {
    if (mkdir(path.c_str(), 0755) == 0) return true;
    return errno == EEXIST;
}

static bool ensure_dir(const string& path) {
    return mkdir_p(path);
}

static string self_dir() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    string exe = buf;
    size_t slash = exe.find_last_of('/');
    return slash == string::npos ? "." : exe.substr(0, slash);
}

static string default_config(const Paths& p, const Model& w, const Model& l) {
    stringstream ss;
    ss << "# dictatd configuration\n"
       << "# Edit this file, then run: dictatd restart\n\n"
       << "whisper_model=" << w.file << "\n"
       << "llama_model=" << l.file << "\n"
       << "whisper_threads=6\n"
       << "llama_threads=2\n"
       << "chunk_seconds=3.0\n\n"
       << "# Models:\n"
       << "#   " << w.display << "  " << kWhisperMin / 1000000 << " MB\n"
       << "#   " << l.display << "   " << kLlamaMin / 1000000 << " MB\n"
       << "# Re-download a missing model with: dictatd setup\n";
    return ss.str();
}

// Returns true if `cmd` ran and exited 0, with stdout/stderr inherited.
static bool run_cmd(const string& cmd) {
    int rc = system(cmd.c_str());
    if (rc == -1) return false;
    return WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
}

static bool have_downloader() {
    return system("command -v curl >/dev/null 2>&1") == 0 ||
           system("command -v wget >/dev/null 2>&1") == 0;
}

// Download `url` to `dest` (resuming any partial file). Returns true on
// success and a non-trivial file size.
static bool download(const string& url, const string& dest, long min_bytes,
                     bool quiet) {
    if (!have_downloader()) {
        fprintf(stderr, "error: neither curl nor wget found\n");
        return false;
    }
    string cmd;
    string prog = "curl";
    bool curl = true;
    if (system("command -v curl >/dev/null 2>&1") != 0) {
        curl = false;
        prog = "wget";
    }
    if (curl) {
        cmd = "curl -L --fail --retry 3 " +
              string(quiet ? "-sS" : "-#") + " -C - -o '" + dest + "' '" + url + "'";
    } else {
        cmd = "wget -c -O '" + dest + "' '" + url + "'" + (quiet ? " -q" : "");
    }
    if (!quiet) fprintf(stderr, "  [downloading %s -> %s]\n", prog.c_str(), dest.c_str());
    if (!run_cmd(cmd)) return false;
    long size = 0;
    if (!file_size(dest, &size)) return false;
    if (size < min_bytes) return false;
    return true;
}

static pid_t read_pid(const string& pidfile) {
    FILE* f = fopen(pidfile.c_str(), "r");
    if (!f) return -1;
    long pid = -1;
    if (fscanf(f, "%ld", &pid) != 1) pid = -1;
    fclose(f);
    return (pid_t)pid;
}

static void write_pid(const string& pidfile, pid_t pid) {
    FILE* f = fopen(pidfile.c_str(), "w");
    if (f) {
        fprintf(f, "%ld\n", (long)pid);
        fclose(f);
    }
}

static void remove_file(const string& path) {
    unlink(path.c_str());
}

static vector<Model> build_models(const Paths& p) {
    Model w{"whisper_model", p.models + "/ggml-base.en.bin", kWhisperUrl,
            "ggml-base.en.bin", kWhisperMin};
    Model l{"llama_model", p.models + "/qwen2.5-0.5b-instruct-q2_k.gguf", kLlamaUrl,
            "qwen2.5-0.5b-instruct-q2_k.gguf", kLlamaMin};
    return {w, l};
}

static int getcfg(const Paths& p, const string& key, const string& fallback) {
    FILE* f = fopen(p.config.c_str(), "r");
    if (!f) return atoi(fallback.c_str());
    char line[1024];
    int val = atoi(fallback.c_str());
    string want = key + "=";
    while (fgets(line, sizeof(line), f)) {
        string s = trim(line);
        if (s.rfind(want, 0) == 0) {
            val = atoi(s.c_str() + want.size());
            break;
        }
    }
    fclose(f);
    return val;
}

static bool getstr(const Paths& p, const string& key, string* out) {
    FILE* f = fopen(p.config.c_str(), "r");
    if (!f) return false;
    char line[1024];
    string want = key + "=";
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        string s = trim(line);
        if (s.rfind(want, 0) == 0) {
            *out = trim(s.c_str() + want.size());
            found = true;
            break;
        }
    }
    fclose(f);
    return found;
}

// --- setup ---------------------------------------------------------------

static int cmd_setup(const Paths& p) {
    printf("dictatd root: %s\n", p.root.c_str());

    if (!ensure_dir(p.root)) return 1;
    if (!ensure_dir(p.models)) return 1;

    auto models = build_models(p);

    // Write a default config if missing.
    if (access(p.config.c_str(), F_OK) != 0) {
        FILE* f = fopen(p.config.c_str(), "w");
        if (f) {
            fputs(default_config(p, models[0], models[1]).c_str(), f);
            fclose(f);
            printf("wrote default config: %s\n", p.config.c_str());
        }
    }

    bool ok = true;
    for (auto& m : models) {
        long size = 0;
        bool present = file_size(m.file, &size) && size >= m.min_bytes;
        if (present) {
            printf("  [ok] %s (%.1f MB)\n", m.display.c_str(), size / 1048576.0);
            continue;
        }
        if (!have_downloader()) {
            fprintf(stderr, "error: %s missing and no curl/wget to fetch it\n",
                    m.display.c_str());
            ok = false;
            continue;
        }
        printf("  ... %s not installed, downloading (~%.0f MB)\n",
               m.display.c_str(), m.min_bytes / 1048576.0);
        if (download(m.url, m.file, m.min_bytes, false)) {
            long got = 0;
            file_size(m.file, &got);
            printf("  [ok] %s (%.1f MB)\n", m.display.c_str(), got / 1048576.0);
        } else {
            fprintf(stderr, "error: download of %s FAILED (partial file kept, "
                    "rerun `dictatd setup` to resume)\n", m.display.c_str());
            ok = false;
        }
    }
    return ok ? 0 : 1;
}

// --- daemon --------------------------------------------------------------

static double getf(const Paths& p, const string& key, double fallback) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3f", fallback);
    return (double)getcfg(p, key, buf);
}

static int start_daemon(const Paths& p, bool print) {
    if (access(p.pipeline.c_str(), X_OK) != 0) {
        fprintf(stderr,
                "error: core binary not found at %s\n"
                "       build it first (see the Makefile)\n", p.pipeline.c_str());
        return 2;
    }
    pid_t existing = read_pid(p.pid);
    if (existing > 0 && is_running(existing)) {
        fprintf(stderr, "dictation daemon already running (pid %ld)\n", (long)existing);
        return 1;
    }
    remove_file(p.pid);

    int rc = cmd_setup(p);
    if (rc != 0) {
        fprintf(stderr, "error: models are not ready; fix the above and retry\n");
        return 2;
    }

    string whisper, llama;
    bool w = getstr(p, "whisper_model", &whisper);
    bool l = getstr(p, "llama_model", &llama);
    if (!w || !l) {
        fprintf(stderr, "error: whisper_model/llama_model missing from %s\n",
                p.config.c_str());
        return 2;
    }
    int wthreads = getcfg(p, "whisper_threads", "6");
    int lthreads = getcfg(p, "llama_threads", "2");
    double chunk = getf(p, "chunk_seconds", 3.0);

    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 2;
    }
    if (child > 0) {
        // Parent: record the pid and report success.
        write_pid(p.pid, child);
        if (print) {
            printf("dictation daemon started (pid %ld)\n"
                   "  log:   %s\n"
                   "  models under %s\n"
                   "  hold Left Shift+S in any field to dictate\n",
                   (long)child, p.log.c_str(), p.models.c_str());
        }
        return 0;
    }

    // Child: detach, redirect stdio to the log, exec the pipeline.
    setsid();
    int logfd = open(p.log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (logfd < 0) logfd = open("/dev/null", O_WRONLY);
    dup2(logfd, 0);
    dup2(logfd, 1);
    dup2(logfd, 2);
    if (logfd > 2) close(logfd);

    vector<string> args_storage;
    args_storage.push_back(p.pipeline);
    args_storage.push_back(whisper);
    args_storage.push_back(llama);
    string st = "--whisper-threads";
    args_storage.push_back(st);
    args_storage.push_back(to_string(wthreads));
    st = "--llama-threads";
    args_storage.push_back(st);
    args_storage.push_back(to_string(lthreads));
    st = "--chunk-seconds";
    args_storage.push_back(st);
    {
        ostringstream oss;
        oss << chunk;
        args_storage.push_back(oss.str());
    }
    vector<char*> argv;
    for (auto& a : args_storage) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    execv(p.pipeline.c_str(), argv.data());
    _exit(127);
}

static int cmd_stop(const Paths& p, bool print) {
    pid_t pid = read_pid(p.pid);
    if (pid <= 0 || !is_running(pid)) {
        if (print) printf("daemon is not running\n");
        remove_file(p.pid);
        return 1;
    }
    if (print) printf("stopping daemon (pid %ld)...\n", (long)pid);
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        if (!is_running(pid)) break;
        usleep(100 * 1000);
    }
    if (is_running(pid)) {
        fprintf(stderr, "daemon did not exit cleanly, sending SIGKILL\n");
        kill(pid, SIGKILL);
    }
    remove_file(p.pid);
    if (print) printf("stopped\n");
    return 0;
}

static int cmd_status(const Paths& p) {
    bool have_core = access(p.pipeline.c_str(), X_OK) == 0;
    pid_t pid = read_pid(p.pid);
    bool running = pid > 0 && is_running(pid);

    printf("dictatd\n");
    printf("  root:       %s\n", p.root.c_str());
    printf("  core:       %s (%s)\n", p.pipeline.c_str(),
           have_core ? "present" : "MISSING - run make");
    printf("  daemon:     %s\n", running ? ("running (pid " + to_string(pid) + ")").c_str()
                                         : "not running");
    for (auto& m : build_models(p)) {
        long size = 0;
        bool present = file_size(m.file, &size);
        if (present && size >= m.min_bytes) {
            printf("  model:      %-32s ok (%.1f MB)\n", m.display.c_str(),
                   size / 1048576.0);
        } else if (present && !file_size(m.file, &size)) {
            printf("  model:      %-32s present but unreadable\n", m.display.c_str());
        } else if (present) {
            printf("  model:      %-32s unfinished (%.1f MB, need %.1f MB)\n",
                   m.display.c_str(), size / 1048576.0, m.min_bytes / 1048576.0);
        } else {
            printf("  model:      %-32s missing (dictatd setup)\n", m.display.c_str());
        }
    }
    return 0;
}

static void cmd_log(const Paths& p) {
    string cmd = "tail -n 50 -F '" + p.log + "' 2>/dev/null || "
                 "echo 'no log yet; start the daemon first'";
    run_cmd(cmd);
}

static int cmd_systemd(const Paths& p) {
    string dir = p.home + "/.config/systemd/user";
    string unit = dir + "/dictatd.service";
    if (!mkdir_p(dir)) {
        fprintf(stderr, "error: cannot create %s\n", dir.c_str());
        return 1;
    }
    FILE* f = fopen(unit.c_str(), "w");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", unit.c_str());
        return 1;
    }
    fprintf(f,
            "[Unit]\n"
            "Description=Dictation daemon (voice_pipeline)\n"
            "After=sound.target\n"
            "\n"
            "[Service]\n"
            "Type=simple\n"
            "ExecStart=%s start\n"
            "ExecStop=%s stop\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            "\n"
            "[Install]\n"
            "WantedBy=default.target\n", p.pipeline.c_str(), p.pipeline.c_str());
    fclose(f);
    printf("wrote %s\n\n"
           "Enable + start now:\n"
           "  systemctl --user daemon-reload\n"
           "  systemctl --user enable --now dictatd\n"
           "  systemctl --user status dictatd\n",
           unit.c_str());
    return 0;
}

int main(int argc, char** argv) {
    Paths p;
    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    p.home = home;
    p.root = p.home + "/.dictatd";
    p.models = p.root + "/models";
    p.config = p.root + "/config";
    p.log = p.root + "/dictatd.log";
    p.pid = p.root + "/dictatd.pid";
    p.self = self_dir();
    p.pipeline = p.self + "/voice_pipeline";

    string cmd = argc > 1 ? string(argv[1]) : "start";
    if (cmd == "setup")          return cmd_setup(p);
    if (cmd == "start")          return start_daemon(p, true);
    if (cmd == "stop")           return cmd_stop(p, true);
    if (cmd == "restart")        { cmd_stop(p, false); return start_daemon(p, true); }
    if (cmd == "status")         return cmd_status(p);
    if (cmd == "log")            { cmd_log(p); return 0; }
    if (cmd == "systemd")        return cmd_systemd(p);
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
        printf("%s <command>\n\n"
               "  setup         generate ~/.dictatd and download missing models\n"
               "  start         start the dictation daemon in the background\n"
               "  stop          stop the daemon\n"
               "  restart       stop, then start\n"
               "  status        daemon + model status\n"
               "  log           tail the daemon log\n"
               "  systemd       write a user systemd unit for the daemon\n\n"
               "  (no command)  operates like `start`\n", argv[0]);
        return 0;
    }
    fprintf(stderr, "unknown command '%s' (see `%s help`)\n", argv[0], cmd.c_str());
    return 1;
}