## Compile commands:

**AES-CUDA.cpp**: nvcc -std=c++20 -x cu -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80 AES-CUDA.cpp -o AES-CUDA -O3 -lcrypto -Xcompiler="-maes"

**AES-VECT.cpp**: g++ -std=c++20 AES-VECT.cpp -o AES-VECT -w -O3 -lcrypto -maes

or

**make vect**  
**make cuda0**  
**make cuda1**  
**make cuda2**

 use the **line** command to compile with the -lineinfo flag for NCU. Ex. make line0
