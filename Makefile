#compiler commands
GPP  := g++
NVCC := nvcc

#targets
TARGET_CPU  := AES-VECT
TARGET_CUDA0 := AES-CUDA-0
TARGET_CUDA1 := AES-CUDA-1
TARGET_CUDA2 := AES-CUDA-2

#source files
SRC_CPU  := AES-VECT.cpp
SRC_CUDA0 := AES-CUDA-0.cpp
SRC_CUDA1 := AES-CUDA-1.cpp
SRC_CUDA2 := AES-CUDA-2.cpp

#compilation flags
CPU_FLAGS := -std=c++20 -w -O3 -maes

CUDA_FLAGS := -std=c++20 -x cu -w -O3 \
              --gpu-architecture=compute_80 \
              --gpu-code=sm_80 \
              -Xcompiler="-maes"
CRYPTO := -lcrypto

CUDA_FLAGS_LINEINFO := $(CUDA_FLAGS) -lineinfo

all: vect cuda0 cuda1 cuda2

#vectorized
vect: $(SRC_CPU)
	$(GPP) $(CPU_FLAGS) $< -o $(TARGET_CPU) $(CRYPTO)

#CUDA0
cuda0: $(SRC_CUDA0)
	$(NVCC) $(CUDA_FLAGS) $< -o $(TARGET_CUDA0) $(CRYPTO)

#CUDA0 with -lineinfo
line0: $(SRC_CUDA0)
	$(NVCC) $(CUDA_FLAGS_LINEINFO) $< -o $(TARGET_CUDA0) $(CRYPTO)

#CUDA1
cuda1: $(SRC_CUDA1)
	$(NVCC) $(CUDA_FLAGS) $< -o $(TARGET_CUDA1) $(CRYPTO)

#CUDA1 with -lineinfo
line1: $(SRC_CUDA1)
	$(NVCC) $(CUDA_FLAGS_LINEINFO) $< -o $(TARGET_CUDA1) $(CRYPTO)

#CUDA2
cuda2: $(SRC_CUDA2)
	$(NVCC) $(CUDA_FLAGS) $< -o $(TARGET_CUDA2) $(CRYPTO)

#CUDA2 with -lineinfo
line2: $(SRC_CUDA2)
	$(NVCC) $(CUDA_FLAGS_LINEINFO) $< -o $(TARGET_CUDA2) $(CRYPTO)



# Clean binaries
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA0) $(TARGET_CUDA1) $(TARGET_CUDA2)

.PHONY: all clean vect cuda0 line0 cuda1 line1 cuda2 line2