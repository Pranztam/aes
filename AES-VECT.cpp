#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <iostream>
#include <openssl/evp.h>
#include "AES.hpp"

constexpr size_t THREAD_COUNT = 32;
constexpr size_t BLOCK_SIZE = 16;

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

//the encryption function will encrypt the chunk using the same technique used in the openssl implementation, which is optimizing the pipeline usage
//by keeping it almost always busy with 8 parallel encryption "streams". when the remainer of the data is not enough to warrant another call to 8-blocks
//the encryption will proceed as normal
void encrypt(unsigned char* data, const size_t size, const unsigned char* nonce, size_t counter, Cipher::Aes<256>& aes) {
    size_t i = 0;

    //if size allows for it, we compute the nonce into the states and proceed to loop remembering that state(i) = nonce | counter(i)
    if(i + 64 <= size){
        unsigned char states[64];
        for (; i + 64 <= size; i += 64) {
            for(int j = 0; j < 4; j++){
                memcpy(states+j*16,nonce,12);
                uint32_t ctr = static_cast<uint32_t>(counter++);
                states[j*16 + 12] = (ctr >> 24) & 0xff;
                states[j*16 + 13] = (ctr >> 16) & 0xff;
                states[j*16 + 14] = (ctr >>  8) & 0xff;
                states[j*16 + 15] = ctr & 0xff;
            }
            aes.encrypt_4_blocks(states);
            for (int j = 0; j < 64; j++)
                data[i + j] ^= states[j];
        }
    }

    if(i < size){
        unsigned char state[16];        
        for (; i < size; i += BLOCK_SIZE) {
            memcpy(state,nonce,12);
            uint32_t ctr = static_cast<uint32_t>(counter++);
            state[12] = (ctr >> 24) & 0xff;
            state[13] = (ctr >> 16) & 0xff;
            state[14] = (ctr >>  8) & 0xff;
            state[15] = ctr & 0xff;
            
            aes.encrypt_block(state);
            for (int j = 0; j < 16; j++)
                data[i + j] ^= state[j];
        }
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

    //nonce generation
    unsigned char nonce[12];
    for (int i = 0; i < 12; i++)
        nonce[i] = static_cast<unsigned char>(dist(gen));

	//the Aes class takes care of the key expansion within its constructor
    Cipher::Aes<256> aes(key);

 
    //key generation for the openssl check
    // AES_KEY enc_key;
    // AES_set_encrypt_key(key, 256, &enc_key);

    std::vector<std::thread> threads;

    size_t chunk_size = SIZE_MB / THREAD_COUNT;
    chunk_size -= (chunk_size % BLOCK_SIZE);

    uint64_t start_time = current_time_nsecs();

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == THREAD_COUNT - 1) ? SIZE_MB : start + chunk_size;

            encrypt(input + start, end - start, nonce, start/16, aes);
        });
    }

    for (auto& th : threads)
        th.join();

    uint64_t end_time = current_time_nsecs();
    std::cout<<"elapsed time: "<< end_time - start_time<<std::endl;

    // for (size_t i = 0; i < SIZE_MB; i += BLOCK_SIZE)
        // AES_encrypt(reference + i, reference + i, &enc_key);
    
    unsigned char iv[16];
    memcpy(iv, nonce, 12);
    iv[12] = iv[13] = iv[14] = iv[15] = 0;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key, iv);
    int outlen;
    EVP_EncryptUpdate(ctx, reference, &outlen, reference, SIZE_MB);
    EVP_CIPHER_CTX_free(ctx);

    //correctness check
    if (std::memcmp(input, reference, SIZE_MB) == 0)
        std::cout << "Encryption correct"<<std::endl;
    else
        std::cout << "Error in the encryption"<<std::endl;

    free(input);
    free(reference);

    return 0;
}