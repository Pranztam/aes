# Makefile

# Compiler commands
GPP  := g++
NVCC := nvcc

# Targets
TARGET_CPU  := AES-VECT
TARGET_CUDA0 := AES-CUDA-0
TARGET_CUDA1 := AES-CUDA-1

# Source files
SRC_CPU  := AES-VECT.cpp
SRC_CUDA0 := AES-CUDA-0.cpp
SRC_CUDA1 := AES-CUDA-1.cpp

CPU_FLAGS := -std=c++20 -w -O3 -lcrypto -maes

CUDA_FLAGS := -std=c++20 -x cu -w -O3 \
              --gpu-architecture=compute_80 \
              --gpu-code=sm_80 \
              -lcrypto \
              -Xcompiler="-maes"

CUDA_FLAGS_LINEINFO := $(CUDA_FLAGS) -lineinfo

all: vect cuda0 cuda1

#vectorized
vect: $(SRC_CPU)
	$(GPP) $(CPU_FLAGS) $< -o $(TARGET_CPU)

#CUDA0
cuda0: $(SRC_CUDA0)
	$(NVCC) $(CUDA_FLAGS) $< -o $(TARGET_CUDA0)

#CUDA0 with -lineinfo
line0: $(SRC_CUDA0)
	$(NVCC) $(CUDA_FLAGS_LINEINFO) $< -o $(TARGET_CUDA0)

#CUDA1
cuda1: $(SRC_CUDA1)
	$(NVCC) $(CUDA_FLAGS) $< -o $(TARGET_CUDA1)

#CUDA1 with -lineinfo
line1: $(SRC_CUDA1)
	$(NVCC) $(CUDA_FLAGS_LINEINFO) $< -o $(TARGET_CUDA1)



# Clean binaries
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA0) $(TARGET_CUDA1)

.PHONY: all clean vect cuda line0