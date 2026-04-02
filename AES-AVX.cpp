#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <iostream>
#include <openssl/aes.h>
#include "AES.hpp"

constexpr size_t THREAD_COUNT = 32;
constexpr size_t BLOCK_SIZE = 16;
constexpr size_t TOTAL_SIZE = 32 * 1024 * 1024;

void encrypt_block(unsigned char* block);

//sort of thread coarsening, each thread will encrypt >=8 block, calling the encrypt_block function, which is defined in AES.hpp, when the thread can't encrypt
//any more chunks it starts processing blocks at a time
inline void encrypt_chunk(unsigned char* data, Cipher::Aes<256>& aes) {
    aes.encrypt_block(data +  0);
    aes.encrypt_block(data + 16);
    aes.encrypt_block(data + 32);
    aes.encrypt_block(data + 48);
    aes.encrypt_block(data + 64);
    aes.encrypt_block(data + 80);
    aes.encrypt_block(data + 96);
    aes.encrypt_block(data + 112);
}

void encrypt(unsigned char* data, size_t size, Cipher::Aes<256>& aes) {
    size_t i = 0;

    for (; i + 128 <= size; i += 128) {
        encrypt_chunk(data + i, aes);
    }

    for (; i < size; i += BLOCK_SIZE) {
        aes.encrypt_block(data + i);
    }
}

unsigned char* read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    // fseek(f, 0, SEEK_END);
    // *size = ftell(f);
    // rewind(f);

    unsigned char* data = (unsigned char*)malloc(TOTAL_SIZE);
    fread(data, 1, TOTAL_SIZE, f);
    fclose(f);

    return data;
}

void write_file(const char* filename, unsigned char* data) {
    FILE* f = fopen(filename, "wb");
    fwrite(data, 1, TOTAL_SIZE, f);
    fclose(f);
}

int main() {
    unsigned char* input = read_file("input.bin");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    //copy of original data to reference array for correctness check
    unsigned char* reference = (unsigned char*)malloc(TOTAL_SIZE);
    memcpy(reference, input, TOTAL_SIZE);

    //key generation
    unsigned char key[32];
    for (int i = 0; i < 32; ++i)
        key[i] = static_cast<unsigned char>(dist(gen));

    Cipher::Aes<256> aes(key);

 
    //key generation for the openssl check
    AES_KEY enc_key;
    AES_set_encrypt_key(key, 256, &enc_key);

    std::vector<std::thread> threads;

    size_t chunk_size = TOTAL_SIZE / THREAD_COUNT;
    chunk_size -= (chunk_size % BLOCK_SIZE);

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == THREAD_COUNT - 1) ? TOTAL_SIZE : start + chunk_size;

            encrypt(input + start, end - start, aes);
        });
    }

    for (auto& th : threads)
        th.join();

    for (size_t i = 0; i < TOTAL_SIZE; i += BLOCK_SIZE)
        AES_encrypt(reference + i, reference + i, &enc_key);

    // Verify
    if (std::memcmp(input, reference, TOTAL_SIZE) == 0)
        std::cout << "Encryption correct"<<std::endl;
    else
        std::cout << "Error in the encryption"<<std::endl;

    free(input);
    free(reference);

    return 0;
}

// srun -p gpu-shared --time 00:00:10 --nodelist=node09 g++ test.cpp -o test -O3 -maes -pthread -lcrypto -std=c++20