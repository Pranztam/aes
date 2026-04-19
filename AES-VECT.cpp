#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <iostream>
#include <openssl/aes.h>
#include "AES.hpp"

constexpr size_t THREAD_COUNT = 32;
constexpr size_t BLOCK_SIZE = 16;

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

void encrypt_block(unsigned char* block);

//each thread will encrypt >=8 block, calling the encrypt_block function, which is defined in AES.hpp, when the thread can't encrypt
//any more chunks it starts processing blocks at a time
inline void encrypt_8_blocks(unsigned char* data, Cipher::Aes<256>& aes) {
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
        encrypt_8_blocks(data + i, aes);
    }

    for (; i < size; i += BLOCK_SIZE) {
        aes.encrypt_block(data + i);
    }
}

int main(int argc, char **argv) {
    if(argc < 2){
        std::cerr<<"Usage: ./AES-VECT <size of text to generate in MB>"<<std::endl;
        return -1;
    }

    if(atoi(argv[1]) == 0 || atoi(argv[1]) > 1024){
        std::cerr<<"Size must be between 1 and 128"<<std::endl;
        return -1;
    }

    size_t SIZE_MB = atoi(argv[1])*1024*1024;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    //plaintext generation
    unsigned char* input = (unsigned char*)malloc(SIZE_MB);
    for (size_t i = 0; i < SIZE_MB; i++)
        input[i] = static_cast<unsigned char>(dist(gen));


    //copy of original data to reference array for correctness check
    unsigned char* reference = (unsigned char*)malloc(SIZE_MB);
    memcpy(reference, input, SIZE_MB);

    //key generation
    unsigned char key[32];
    for (int i = 0; i < 32; ++i)
        key[i] = static_cast<unsigned char>(dist(gen));

	//the Aes class takes care of the key expansion within its constructor
    Cipher::Aes<256> aes(key);

 
    //key generation for the openssl check
    AES_KEY enc_key;
    AES_set_encrypt_key(key, 256, &enc_key);

    std::vector<std::thread> threads;

    size_t chunk_size = SIZE_MB / THREAD_COUNT;
    chunk_size -= (chunk_size % BLOCK_SIZE);

    uint64_t start_time = current_time_nsecs();

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == THREAD_COUNT - 1) ? SIZE_MB : start + chunk_size;

            encrypt(input + start, end - start, aes);
        });
    }

    for (auto& th : threads)
        th.join();

    uint64_t end_time = current_time_nsecs();
    std::cout<<"elapsed time: "<< end_time - start_time<<std::endl;

    for (size_t i = 0; i < SIZE_MB; i += BLOCK_SIZE)
        AES_encrypt(reference + i, reference + i, &enc_key);

    //correctness check
    if (std::memcmp(input, reference, SIZE_MB) == 0)
        std::cout << "Encryption correct"<<std::endl;
    else
        std::cout << "Error in the encryption"<<std::endl;

    free(input);
    free(reference);

    return 0;
}