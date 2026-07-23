#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

bool is_keyboard(int fd) {

    unsigned long evbit = 0;

    if (ioctl(fd, EVIOCGBIT(0, sizeof(evbit)), &evbit) < 0) {
        return false;
    }

    return evbit & (1UL << EV_KEY);
}

int main() {

    vector<int> fds;
    int stream_pipe = -1;
    pid_t stream_pid = -1;

    // Find all keyboards
    DIR* dir = opendir("/dev/input");
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {

        if (strncmp(entry->d_name, "event", 5) != 0) {
            continue;
        }

        string path = "/dev/input/" + string(entry->d_name);
        int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);

        if (fd < 0) {
            continue;
        }

        if (is_keyboard(fd)) {
            cout << "Monitoring: " << path << "\n";
            fds.push_back(fd);
        } else {
            close(fd);
        }
    }

    closedir(dir);

    if (fds.empty()) {
        cerr << "No keyboards found\n";
        return 1;
    }

    bool super_down = false;
    bool running = true;
    string transcription;

    cout << "\nPress Win + Space to start/stop streaming\n";
    cout << "Press Ctrl+C to exit\n\n";

    while (running) {

        for (int fd : fds) {

            struct input_event ev;

            while (read(fd, &ev, sizeof(ev)) == sizeof(ev)) {

                if (ev.type != EV_KEY) {
                    continue;
                }

                if (ev.code == KEY_LEFTMETA || ev.code == KEY_RIGHTMETA) {
                    super_down = (ev.value == 1 || ev.value == 2);
                }

                if (ev.code == KEY_SPACE && ev.value == 1 && super_down) {

                    if (stream_pid < 0) {

                        // Start streaming
                        int pipefd[2];
                        pipe(pipefd);

                        stream_pid = fork();

                        if (stream_pid == 0) {
                            // Child process
                            close(pipefd[0]);
                            dup2(pipefd[1], STDOUT_FILENO);
                            close(pipefd[1]);

                            execl(
                                "../exploration/whisper.cpp/build/bin/whisper-stream",
                                "whisper-stream",
                                "-m", "../exploration/whisper.cpp/models/ggml-base.en.bin",
                                "--step", "500",
                                "--length", "3000",
                                NULL
                            );

                            _exit(1);
                        }

                        // Parent process
                        close(pipefd[1]);
                        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);
                        stream_pipe = pipefd[0];

                        transcription.clear();

                        cout << "\n[STREAM STARTED]\n";

                    } else {

                        // Stop streaming
                        kill(stream_pid, SIGTERM);
                        waitpid(stream_pid, NULL, 0);
                        close(stream_pipe);

                        stream_pid = -1;
                        stream_pipe = -1;

                        cout << "\n[STREAM STOPPED]\n";
                        cout << "\nFull transcription:\n" << transcription << "\n\n";

                        // Use 'transcription' variable here for grammar, LLM, etc.
                    }
                }
            }
        }

        // Read whisper output from pipe
        if (stream_pipe >= 0) {

            char buf[256];
            int n = read(stream_pipe, buf, sizeof(buf) - 1);

            if (n > 0) {
                buf[n] = '\0';
                transcription += buf;
                cout << buf;
                cout.flush();
            }
        }

        usleep(5000);
    }

    for (int fd : fds) {
        close(fd);
    }

    return 0;
}
