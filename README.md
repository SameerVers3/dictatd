# dictatd

Hold **Left Shift+S** in any field and talk, speech is transcribed by
[whisper.cpp](https://github.com/ggerganov/whisper.cpp), grammar-corrected by
[llama.cpp](https://github.com/ggerganov/llama.cpp), and pasted into the
focused window in realtime as you speak.

## Features

- Realtime streaming: each sentence is pasted the moment it's ready
- Grammar and spelling correction via a smollm (will make it optional in the future)
- Works in any field ( wayland/hyprland only ) using a virtual keyboard and clipboard
- `dictatd` launcher handles model install + background daemon
- Configurable threads / chunk size, optional systemd autostart

## Installation

### Dependencies (Arch)

```sh
sudo pacman -S base-devel cmake git curl
# voice input (system microphone)
sudo pacman -S alsa-utils     # arecord (or ffmpeg)
# wayland clipboard + /dev/uinput permissions
sudo pacman -S wl-clipboard
```

### Build

```sh
./build_deps.sh        # clones and builds whisper.cpp & llama.cpp, then makes voice_pipeline
```

`build_deps.sh` builds for the **same architecture** the binaries run on. For
a lean/smaller build you can tweak the `CMAKE_...` flags inside it.

### Install

```sh
ln -s "$PWD/dictatd"         ~/.local/bin/dictatd
ln -s "$PWD/voice_pipeline"  ~/.local/bin/voice_pipeline
```

## Usage

```sh
dictatd setup      # download missing models into ~/.dictatd (resumable)
dictatd start      # run the daemon in the background
dictatd status     # daemon + model status
dictatd stop
dictatd restart
dictatd log        # tail the daemon log
dictatd systemd    # optional: write a systemd user unit
```

then focus any text field, press and hold **Left Shift+S**, speak, and release
when done.

### Configuration (`~/.dictatd/config`)

```
whisper_threads=6
llama_threads=2
chunk_seconds=3.0
```

Models are downloaded into `~/.dictatd/models` on first `setup` and skipped on
subsequent runs if already installed.

## Notes

- The daemon grabs your keyboards while running; keys are re-emitted through a
  virtual keyboard, so the `Left Shift+S` combo never reaches your apps.
- Lower `chunk_seconds` = lower first-word latency. Raise threads if you have
  more CPU cores than the defaults.
