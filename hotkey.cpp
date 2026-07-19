#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

bool is_keyboard(int fd) {

    unsigned long evbit = 0;

    int ok = ioctl(fd, EVIOCGBIT(0, sizeof(evbit)), &evbit); 


    if (ok < 0) {
        return false;
    }

    int has_key = evbit & (1UL << EV_KEY);

    return has_key;
}

int main() {

    vector<int> fds;

    // opening /dev/input directory
    DIR* dir = opendir("/dev/input");
    
    if (!dir) {
        perror("opendir");
        return 1;
    }

    // find all event devices
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {

        if (strncmp(entry->d_name, "event", 5) != 0) {
            continue;
        }

        string path = "/dev/input/" + string(entry->d_name);

        // opening device for reading
        int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);

        if (fd < 0) {
            continue;
        }

        // If it can send keys, monitor it
        if (is_keyboard(fd)) {

            cout << "monitoring: " << path << "\n";
            fds.push_back(fd);

        } else {
            close(fd);
        }
    }

    closedir(dir);

    if (fds.empty()) {
        cerr << "no keyboards found\n";
        return 1;
    }

    // is key down or not
    bool super_down = false;

    bool running = true;
    while (running) {
        for (int fd : fds) {

            struct input_event ev;

            ssize_t ev_size = sizeof(ev);
            while (read(fd, &ev, ev_size) == ev_size) {

                // only key press or relase
                if (ev.type != EV_KEY) {
                    continue;
                }

                // updateing superkey
                if (ev.code == KEY_LEFTMETA || ev.code == KEY_RIGHTMETA) {
                    super_down = (ev.value == 1 || ev.value == 2);
                }

                // win + space combo
                if (ev.code == KEY_SPACE && ev.value == 1 && super_down) {
                    
                    // trigger here all other pipeline
                    cout << "Win + Space pressed!\n";
                }
            }
        }

        usleep(5000);
    }

    for (int fd : fds) {
        close(fd);
    }

    return 0;
}
