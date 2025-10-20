# Compilers
CXX = g++
CC = gcc
NVCC = nvcc

# Paths (absolute path)
# use ta's version
LODEPNG_DIR = /work/b10502010/pp25/hw3/lodepng
GLM_DIR = /work/b10502010/pp25/hw3/glm

# Compiler flags
CPPFLAGS = -I$(LODEPNG_DIR) -I$(GLM_DIR)/include
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70 -Xptxas=-v

# Source files
LODEPNG = $(LODEPNG_DIR)/lodepng.cpp

# Targets
TARGET_CUDA = hw3
TARGET_CPU = hw3_cpu

# CUDA version
all: $(TARGET_CUDA)

$(TARGET_CUDA): hw3.cu $(LODEPNG)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) hw3.cu $(LODEPNG) -o $(TARGET_CUDA)

# CPU version
cpu: $(TARGET_CPU)

$(TARGET_CPU): hw3_cpu.cpp $(LODEPNG)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) hw3_cpu.cpp $(LODEPNG) -o $(TARGET_CPU)

# Clean
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA)
