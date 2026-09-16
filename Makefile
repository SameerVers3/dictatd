CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall

WHISPER_DIR := ../exploration/whisper.cpp
WHISPER_INCLUDES := -I$(WHISPER_DIR)/include -I$(WHISPER_DIR)/ggml/include
WHISPER_LIBS := $(WHISPER_DIR)/build/bin/libwhisper.so

LLAMA_DIR := ../exploration/llama.cpp
LLAMA_INCLUDES := -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/src
LLAMA_LIBS := $(LLAMA_DIR)/build/bin/libllama.so $(LLAMA_DIR)/build/bin/libggml.so $(LLAMA_DIR)/build/bin/libggml-base.so $(LLAMA_DIR)/build/bin/libggml-cpu.so

RPATH := -Wl,-rpath,'$(WHISPER_DIR)/build/bin:$(LLAMA_DIR)/build/bin'
LDFLAGS := -lpthread -ldl -lm
INCLUDES := $(WHISPER_INCLUDES) $(LLAMA_INCLUDES)

TARGET := voice_pipeline
TARGETS := $(TARGET) dictatd
SRCS := main.cpp logger.cpp timer.cpp audio.cpp transcribe.cpp grammar.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean check help

all: $(TARGETS)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(WHISPER_LIBS) $(LLAMA_LIBS) $(RPATH) $(LDFLAGS)

dictatd: dictatd.o
	$(CXX) dictatd.o -o dictatd

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGETS) $(OBJS)

