#include <vector>
#include <thread>
#include <random>
#include <filesystem>
#include <algorithm>
#include <span>
#include <array>
#include <cstring>
#include <iostream>
#include <fstream>
#include <openssl/evp.h>
#include "AES.hpp"

constexpr int THREAD_COUNT = 32;
constexpr int BLOCK_SIZE = 16;

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

using byte = unsigned char;

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

//the encryption function will encrypt the chunk using the same technique used in the openssl implementation, which is optimizing the pipeline usage
//by keeping it almost always busy with 8 parallel encryption "streams". when the remainer of the data is not enough to warrant another call to 8-blocks
//the encryption will proceed as normal
void encrypt(std::span<byte> data, std::span<const byte> nonce, size_t counter, Cipher::Aes<256>& aes) {
    size_t i = 0;

    //if size allows for it, we compute the nonce into the states and proceed to loop remembering that state(i) = nonce | counter(i)
    if(i + 64 <= data.size()){
        std::array<byte, 64> states;
        for (; i + 64 <= data.size(); i += 64) {
            for(int j = 0; j < 4; j++){
                memcpy(states.data() +j*16, nonce.data(), 12);
                uint32_t ctr = static_cast<uint32_t>(counter++);
                states[j*16 + 12] = (ctr >> 24) & 0xff;
                states[j*16 + 13] = (ctr >> 16) & 0xff;
                states[j*16 + 14] = (ctr >>  8) & 0xff;
                states[j*16 + 15] = ctr & 0xff;
            }
            aes.encrypt_4_blocks(states.data());
            for (int j = 0; j < 64; j++)
                data[i + j] ^= states[j];
        }
    }

    if(i < data.size()){
        std::array<byte, 16> state;
        for (; i < data.size(); i += BLOCK_SIZE) {
            memcpy(state.data(), nonce.data(), 12);
            uint32_t ctr = static_cast<uint32_t>(counter++);
            state[12] = (ctr >> 24) & 0xff;
            state[13] = (ctr >> 16) & 0xff;
            state[14] = (ctr >>  8) & 0xff;
            state[15] = ctr & 0xff;
            
            aes.encrypt_block(state.data());
            for (int j = 0; j < 16; j++)
                data[i + j] ^= state[j];
        }
    }
}

int main(int argc, char **argv) {
    if(argc < 2){
        std::cerr<<"Usage: ./AES-CUDA-1 --file <filename> or ./AES-CUDA-1 <size of text to generate in MB>"<<std::endl;
        return -1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<byte> plaintext;

    if (strcmp(argv[1], "--file") == 0) {
        if (argc < 3) {
            std::cerr << "Error: --file requires a filename" << std::endl;
            return -1;
        }

        //plaintext from a file
        plaintext = read_file(argv[2]);
        if (plaintext.empty()) return -1;

    } else {

        //plaintext generated with a given size
        if (atoi(argv[1]) <= 0 || atoi(argv[1]) > 1024) {
            std::cerr << "Size must be between 1 and 1024 MB" << std::endl;
            return -1;
        }

        plaintext.resize(atoi(argv[1])*1024*1024);
        std::generate(plaintext.begin(), plaintext.end(), [&]() { return static_cast<byte>(dist(gen)); });
    }

    //save original plain text in reference before copying the encrypted data back in plaintext
    std::vector<byte> reference = plaintext;

    //key and nonce generation
    std::vector<byte> key(32);
    std::vector<byte> nonce(12);

    std::generate(key.begin(), key.end(), [&]() { return static_cast<byte>(dist(gen)); });
    std::generate(nonce.begin(), nonce.end(), [&]() { return static_cast<byte>(dist(gen)); });

	//here we call the constructor of the class cipher, included in AES.hpp, which transforms the key passed as a parameter
	//into an expanded key of 240 bits
    Cipher::Aes<256> aes(key.data());

    std::vector<std::thread> threads;

    //calculating how much data a thread will encrypt
    size_t chunk_size = plaintext.size() / THREAD_COUNT;
    chunk_size -= (chunk_size % BLOCK_SIZE);

    uint64_t start_time = current_time_nsecs();

    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = (t == THREAD_COUNT - 1) ? plaintext.size() : start + chunk_size;

            encrypt(std::span<byte>(plaintext.data() + start, end - start), std::span<const byte>(nonce), start/16, aes);
        });
    }

    for (auto& th : threads)
        th.join();

    uint64_t end_time = current_time_nsecs();
    std::cout<<"elapsed time: "<< end_time - start_time<<std::endl;

    // std::ofstream file("measurements.txt", std::ios::app); // append mode
    // if (file.is_open())
    //     file << end_time - start_time << "\n";
    
    //correctness check using OpenSSL library functions
    byte iv[16];
    memcpy(iv, nonce.data(), 12);
    iv[12] = iv[13] = iv[14] = iv[15] = 0;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), NULL, key.data(), iv);

    int outlen;
    EVP_EncryptUpdate(ctx, reference.data(), &outlen, reference.data(), reference.size());
    EVP_CIPHER_CTX_free(ctx);

    if (memcmp(plaintext.data(), reference.data(), plaintext.size()) == 0)
        std::cout << "Encryption correct"<<std::endl;
    else
        std::cout << "Error in the encryption"<<std::endl;

    return 0;
}