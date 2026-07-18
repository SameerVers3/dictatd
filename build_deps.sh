#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "building dependencies"
echo ""

# whisper.cpp
WHISPER_DIR="$PARENT_DIR/exploration/whisper.cpp"

if [ ! -d "$WHISPER_DIR" ]; then
    echo "Cloning whisper.cpp..."
    mkdir -p "$PARENT_DIR/exploration"
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
fi

echo "building whisper.cpp"

cd "$WHISPER_DIR"

if [ ! -d build ]; then
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF
fi

cmake --build build -j$(nproc)

echo "whisper.cpp built done."
echo ""

# llama.cpp
LLAMA_DIR="$PARENT_DIR/exploration/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    mkdir -p "$PARENT_DIR/exploration"
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
fi

echo "building llama.cpp..."
cd "$LLAMA_DIR"

if [ ! -d build ]; then
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF
fi

cmake --build build -j$(nproc)

echo "llama.cpp built successfully."
echo ""

echo "build voice_pipeline"
make

echo ""
echo "fone"
echo "Run with:"
echo "  ./voice_pipeline ../exploration/whisper.cpp/models/ggml-base.en.bin ../exploration/llama.cpp/models/Qwen2.5-0.5B-Instruct.Q2_K.gguf 5"
