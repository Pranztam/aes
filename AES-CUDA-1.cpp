#include <iostream>
#include <random>
#include <cstring>
#include <openssl/evp.h>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include "AES.hpp"

#define AES256_ROUNDS 14
#define EXPANDED_KEY_SIZE 240
#define REPLICAS 4

constexpr size_t BLOCK_SIZE = 16;

using byte = unsigned char;

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

//function to read from a file with arbitrary length, adding padding if necessary (block size is 16 byte, the total size must be a multiple)
std::vector<byte> read_file(const std::string& filename) {

    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: file not found: " << filename << std::endl;
        return {};
    }

    auto file_size = std::filesystem::file_size(filename);
    if (file_size == 0) {
        std::cerr << "Error: file is empty: " << filename << std::endl;
        return {};
    }
    size_t padded_size = file_size + (16 - file_size % 16) % 16;

    //we're padding with 0 if needed
    std::vector<byte> buffer(padded_size, 0);
    
    std::ifstream f(filename, std::ios::binary);
    f.read(reinterpret_cast<char*>(buffer.data()), file_size);

    std::cout << "Read " << file_size << " bytes from " << filename << ", padded to " << padded_size << std::endl;

    return buffer;
}

__constant__ byte roundKeys[EXPANDED_KEY_SIZE];
__constant__ byte nonce[12];
__device__ byte d_sbox[256];

byte h_sbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

//gmul is an implementation of multiplication in GF(2^8) finite field, needed to compute the t-tables.
//more specifically, the only values of b are either 2 or 3, because in the mixcolumns matrix we only have '1','2','3' multipliers, as it can be seen from the the AES NIST standard paper.
//gmul performs a multiplication in the GF(2^8) finite field. This particular implementation is called "Russian peasant" -> https://en.wikipedia.org/wiki/Finite_field_arithmetic#C_programming_example
byte gmul(byte a, byte b) {
    byte p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        if (a & 0x80)
            a = (a << 1) ^ 0x11b;
        else
            a <<= 1;
        
        b >>= 1;
    }
    return p;
}

//rotate left of a 32-bit word by 1 byte
uint32_t rotate_word(uint32_t x) {
    return (x >> 8) | (x << 24);
}

//creation of the 4-tables
void t_tables(uint4* T) {

    for (int idx = 0; idx < 256; idx++) {
        byte s  = h_sbox[idx];
        byte s2 = gmul(s, 2);
        byte s3 = gmul(s, 3);
        uint32_t t  = (s2 << 24) | (s << 16) | (s << 8) | s3;
        uint32_t t1 = rotate_word(t);
        uint32_t t2 = rotate_word(t1);
        uint32_t t3 = rotate_word(t2);

        // XOR-permuted position
        // int row = idx >> 5;
        // int col = (idx & 31) ^ row;
        T[idx] = {t, t1, t2, t3};        

    }
}

//last round (14th) of the 256 bit encryption. It requires a separate computation because it does not perform a mixcolumn operation
//for simplicity of writing, it has been separated from the main rounds computation where we can use bytes instead of 32 bit words as units.
__device__ void final_round(uint32_t* state, const byte* roundKey, const byte* s_sbox);

//main encrypting function (rounds). Note that even though the state of AES is a column oriented data structure, we can still reason in a row-like manner:
//data input: b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 -> b0 b1 b2 b3 create a column, but in our case we receive them sequentially and therefore can treat them as a row.
__global__ void aes256_kernel(byte* data, size_t numBlocks, uint4* d_T){
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool thread_bound_check = (tid + 7 * stride) < numBlocks;

    __shared__ uint32_t T0[256];
    __shared__ uint32_t T1[256];
    __shared__ uint32_t T2[256];
    __shared__ uint32_t T3[256];
    __shared__ byte s_sbox[256];

    //in order to remove uncoalesced global accesses we use uint4 variables, letting us move 128bits in one swoop.
    //the x,y,z,w indexes contain, in order, the t-tables from T0 to T1
    uint4 table_entry = d_T[threadIdx.x];
    T0[threadIdx.x] = table_entry.x;
    T1[threadIdx.x] = table_entry.y;
    T2[threadIdx.x] = table_entry.z;
    T3[threadIdx.x] = table_entry.w;
    s_sbox[threadIdx.x] = d_sbox[threadIdx.x];

    __syncthreads();

    //we want to work on 8 blocks per thread. Each thread encrypts 8 blocks in a stride pattern.
    for (int b = 0; b < 8; b++) {
        size_t idx = tid + b * stride;
        if (!thread_bound_check && idx >= numBlocks) return;

        //creating the state that will be encrypted. It contains nonce || ctr. In our case idx is a perfect ctr
        uint32_t state[4];  
        for(int i = 0; i < 3; i++)
            state[i] = (nonce[i*4+0] << 24) | (nonce[i*4+1] << 16) | (nonce[i*4+2] << 8) | (nonce[i*4+3]);
        state[3] = static_cast<uint32_t>(idx); 
        

        //before the regular rounds, we perform the first add round key after we compact the expanded key in 32-bit words (in regular rounds, the add round key operation does the same thing
        //except in final round, whose unit of work is bytes instead of 32-bit words. See final_round() above)
        for (int i = 0; i < 4; i++) {
            uint32_t comp_k = (roundKeys[i*4+0] << 24) | (roundKeys[i*4+1] << 16) | (roundKeys[i*4+2] << 8) | (roundKeys[i*4+3]);
            state[i] ^= comp_k;
        }

        //main rounds, which perform SubBytes, ShiftRows and MixColumns thanks to the t-tables, whose entries are 32-bits words each and contain 256 elements.
	    //more precisely the SubBytes and MixColumns are integrated in the t-tables creation, as we've seen above, whilst now we are performing a
	    //ShiftRows operation by manipulating the state array indexes in the assignments

	    uint32_t t_table[4];
        for (int round = 1; round < AES256_ROUNDS; round++) {

            t_table[0] = T0[(state[0] >> 24) & 0xff] ^ T1[(state[1] >> 16) & 0xff] ^ T2[(state[2] >> 8) & 0xff] ^ T3[(state[3]) & 0xff];
            t_table[1] = T0[(state[1] >> 24) & 0xff] ^ T1[(state[2] >> 16) & 0xff] ^ T2[(state[3] >> 8) & 0xff] ^ T3[(state[0]) & 0xff];
            t_table[2] = T0[(state[2] >> 24) & 0xff] ^ T1[(state[3] >> 16) & 0xff] ^ T2[(state[0] >> 8) & 0xff] ^ T3[(state[1]) & 0xff];
            t_table[3] = T0[(state[3] >> 24) & 0xff] ^ T1[(state[0] >> 16) & 0xff] ^ T2[(state[1] >> 8) & 0xff] ^ T3[(state[2]) & 0xff];

            //add round key
            for (int i = 0; i < 4; i++) {
                uint32_t comp_k = (roundKeys[round*16 + i*4+0] << 24) | (roundKeys[round*16 + i*4+1] << 16) | (roundKeys[round*16 + i*4+2] << 8) | (roundKeys[round*16 + i*4+3]);
	    		t_table[i] ^= comp_k;
            }

            //finally we recreate the state from the t-tables
            state[0] = t_table[0];
            state[1] = t_table[1];
            state[2] = t_table[2];
            state[3] = t_table[3];

        }

        //perform final round (see final_round()) above
        final_round(state, roundKeys + AES256_ROUNDS * 16, s_sbox);

        //positioning the thread on the correct data index based on its identifier
        //in order to remove uncoalesced global accesses we use uint4 variables, letting us move 128bits in one swoop.
        uint4* out = reinterpret_cast<uint4*>(data + idx * 16);

        //before storing the data back we change its endianness to match the hosts'. In the normal version we don't need to do this because we store byte by byte. 
        out->x ^= __byte_perm(state[0], 0, 0x0123);
        out->y ^= __byte_perm(state[1], 0, 0x0123);
        out->z ^= __byte_perm(state[2], 0, 0x0123);
        out->w ^= __byte_perm(state[3], 0, 0x0123);
    }
}

__device__ void final_round(uint32_t* state, const byte* roundKey, const byte* s_sbox) {
    byte s[16];

    //we don't need the state represented as 4 32-bit words, so we extract each byte
    for (int i = 0; i < 4; i++) {
        s[i*4+0] = (state[i] >> 24) & 0xff;
        s[i*4+1] = (state[i] >> 16) & 0xff;
        s[i*4+2] = (state[i] >> 8) & 0xff;
        s[i*4+3] = (state[i]) & 0xff;
    }

    byte temp_state[16];

    //here we perform a SubBytes and ShiftRows operation simultaneously. Instead of assigning the value of the shifted row, we use corresponding sbox value.
    //0  4  8  12        0  4  8  12
    //1  5  9  13   ---> 5  9  13 1
    //2  6  10 14        10 14 2  6
    //3  7  11 15        15 3  7  11
    temp_state[0]  = s_sbox[s[0]];
    temp_state[1]  = s_sbox[s[5]];
    temp_state[2]  = s_sbox[s[10]];
    temp_state[3]  = s_sbox[s[15]];

    temp_state[4]  = s_sbox[s[4]];
    temp_state[5]  = s_sbox[s[9]];
    temp_state[6]  = s_sbox[s[14]];
    temp_state[7]  = s_sbox[s[3]];

    temp_state[8]  = s_sbox[s[8]];
    temp_state[9]  = s_sbox[s[13]];
    temp_state[10] = s_sbox[s[2]];
    temp_state[11] = s_sbox[s[7]];

    temp_state[12] = s_sbox[s[12]];
    temp_state[13] = s_sbox[s[1]];
    temp_state[14] = s_sbox[s[6]];
    temp_state[15] = s_sbox[s[11]];

    //adding round key
    for (int i = 0; i < 16; i++)
        temp_state[i] ^= roundKey[i];

    //recreating the 4 32-bit words comprising the state
    for (int i = 0; i < 4; i++)
        state[i] = (temp_state[i*4+0] << 24) | (temp_state[i*4+1] << 16) | (temp_state[i*4+2] << 8) | (temp_state[i*4+3]);
}

int main(int argc, char** argv) {
    if(argc < 2){
        std::cerr<<"Usage: ./AES-CUDA-1 --file <filename> or ./AES-CUDA-1 <size of text to generate in MB>"<<std::endl;
        return -1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<byte> h_data;

    if (strcmp(argv[1], "--file") == 0) {
        if (argc < 3) {
            std::cerr << "Error: --file requires a filename" << std::endl;
            return -1;
        }

        //plaintext from a file
        h_data = read_file(argv[2]);
        if (h_data.empty()) return -1;

    } else {

        //plaintext generated with a given size
        if (atoi(argv[1]) == 0 || atoi(argv[1]) > 1024) {
            std::cerr << "Size must be between 1 and 1024 MB" << std::endl;
            return -1;
        }

        h_data.resize(atoi(argv[1])*1024*1024);
        std::generate(h_data.begin(), h_data.end(), [&]() { return static_cast<byte>(dist(gen)); });
    }

    //save original plain text in reference before copying the encrypted data back in h_data
    std::vector<byte> reference = h_data;
    size_t numBlocks = h_data.size() / 16;

    //key and nonce generation
    std::vector<byte> key(32);
    std::vector<byte> h_nonce(12);

    std::generate(key.begin(), key.end(), [&]() { return static_cast<byte>(dist(gen)); });
    std::generate(h_nonce.begin(), h_nonce.end(), [&]() { return static_cast<byte>(dist(gen)); });

	//here we call the constructor of the class cipher, included in AES.hpp, which transforms the key passed as a parameter
	//into an expanded key of 240 bits
    Cipher::Aes<256> aes(key.data());

    //t-tables creation
    uint4 h_T[256];
    t_tables(h_T);

    uint4* d_T;

    int threads = 256;
    int blocks = (numBlocks/8 + threads - 1) / threads;

    byte *d_data, *d_keys, *d_nonce;

    uint64_t start_time = current_time_nsecs();

    gpuErrchk(cudaMalloc(&d_data, h_data.size()));
    gpuErrchk(cudaMemcpy(d_data, h_data.data(), h_data.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(roundKeys, aes.getRoundKeys(), EXPANDED_KEY_SIZE));
    gpuErrchk(cudaMemcpyToSymbol(nonce, h_nonce.data(), 12));
    gpuErrchk(cudaMemcpyToSymbol(d_sbox, h_sbox, 256));
    
    gpuErrchk(cudaMalloc(&d_T, 256 * sizeof(uint4)));
    gpuErrchk(cudaMemcpy(d_T, h_T, 256 * sizeof(uint4), cudaMemcpyHostToDevice));

    aes256_kernel<<<blocks, threads>>>(d_data, numBlocks, d_T);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_data.data(), d_data, h_data.size(), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    std::cout<<"elapsed time: "<< end_time - start_time<<std::endl;

    // std::ofstream file("measurements.txt", std::ios::app);
    // if (file.is_open())
    //     file << end_time - start_time << "\n";

    //correctness check comparing the original data array and a reference array encrypted using an OpenSSL library function
    byte iv[16];
    memcpy(iv, h_nonce.data(), 12);
    iv[12] = iv[13] = iv[14] = iv[15] = 0;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key.data(), iv);

    int outlen;
    EVP_EncryptUpdate(ctx, reference.data(), &outlen, reference.data(), reference.size());
    EVP_CIPHER_CTX_free(ctx);

    if (memcmp(h_data.data(), reference.data(), h_data.size()) == 0)
        std::cout << "Encryption correct"<<std::endl;
    else
        std::cout << "Error in the encryption"<<std::endl;
    
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_T));

    return 0;
}