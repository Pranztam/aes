#include <iostream>
#include <limits>
#include <climits>
#include <cstring>
#include <exception>

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace Cipher {
  template <size_t key_bits = 256>
  class Aes {
    static constexpr size_t AES_BLOCK = 16;
    static constexpr size_t Nb = 4;
    static constexpr size_t Nk = key_bits / 32;
    static constexpr size_t Nr = Nk + 6;
    static constexpr size_t round_keys_size = 4 * Nb * (Nr + 1);
    static constexpr size_t key_bytes = key_bits / 8;

    unsigned char round_keys[round_keys_size];

    inline __m128i AES_128_ASSIST(__m128i tmp1, __m128i tmp2) {
      __m128i tmp3;
      tmp2 = _mm_shuffle_epi32(tmp2, 0xff);
      tmp3 = _mm_slli_si128(tmp1, 0x4);
      tmp1 = _mm_xor_si128(tmp1, tmp3);
      tmp3 = _mm_slli_si128(tmp3, 0x4);
      tmp1 = _mm_xor_si128(tmp1, tmp3);
      tmp3 = _mm_slli_si128(tmp3, 0x4);
      tmp1 = _mm_xor_si128(tmp1, tmp3);
      tmp1 = _mm_xor_si128(tmp1, tmp2);
      return tmp1;
    }

    void AES_128_Key_Expansion(const unsigned char *user_key, unsigned char *key) {
      __m128i tmp1, tmp2;
      __m128i *key_sched = (__m128i *) key;

      tmp1 = _mm_loadu_si128((__m128i *) user_key);
      key_sched[0] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x1);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[1] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x2);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[2] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x4);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[3] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x8);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[4] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x10);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[5] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x20);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[6] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x40);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[7] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x80);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[8] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x1b);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[9] = tmp1;
      tmp2 = _mm_aeskeygenassist_si128(tmp1, 0x36);
      tmp1 = AES_128_ASSIST(tmp1, tmp2);
      key_sched[10] = tmp1;
    }

    inline void KEY_192_ASSIST(__m128i *tmp1, __m128i *tmp2, __m128i *tmp3) {
      __m128i tmp4;
      *tmp2 = _mm_shuffle_epi32(*tmp2, 0x55);
      tmp4 = _mm_slli_si128(*tmp1, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      *tmp1 = _mm_xor_si128(*tmp1, *tmp2);
      *tmp2 = _mm_shuffle_epi32(*tmp1, 0xff);
      tmp4 = _mm_slli_si128(*tmp3, 0x4);
      *tmp3 = _mm_xor_si128(*tmp3, tmp4);
      *tmp3 = _mm_xor_si128(*tmp3, *tmp2);
    }

    void AES_192_Key_Expansion(const unsigned char *user_key, unsigned char *key) {
      __m128i tmp1, tmp2, tmp3;
      __m128i *key_sched = (__m128i *) key;
      tmp1 = _mm_loadu_si128((__m128i *) user_key);
      tmp3 = _mm_loadu_si128((__m128i *) (user_key + 16));
      key_sched[0] = tmp1;
      key_sched[1] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x1);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[1] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(key_sched[1]), _mm_castsi128_pd(tmp1), 0));
      key_sched[2] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp3), 1));
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x2);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[3] = tmp1;
      key_sched[4] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x4);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[4] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(key_sched[4]), _mm_castsi128_pd(tmp1), 0));
      key_sched[5] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp3), 1));
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x8);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[6] = tmp1;
      key_sched[7] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x10);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[7] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(key_sched[7]), _mm_castsi128_pd(tmp1), 0));
      key_sched[8] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp3), 1));
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x20);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[9] = tmp1;
      key_sched[10] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x40);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[10] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(key_sched[10]), _mm_castsi128_pd(tmp1), 0));
      key_sched[11] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp3), 1));
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x80);
      KEY_192_ASSIST(&tmp1, &tmp2, &tmp3);
      key_sched[12] = tmp1;
    }

    inline void KEY_256_ASSIST_1(__m128i *tmp1, __m128i *tmp2) {
      __m128i tmp4;
      *tmp2 = _mm_shuffle_epi32(*tmp2, 0xff);
      tmp4 = _mm_slli_si128(*tmp1, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp1 = _mm_xor_si128(*tmp1, tmp4);
      *tmp1 = _mm_xor_si128(*tmp1, *tmp2);
    }

    inline void KEY_256_ASSIST_2(__m128i *tmp1, __m128i *tmp3) {
      __m128i tmp2, tmp4;
      tmp4 = _mm_aeskeygenassist_si128(*tmp1, 0x0);
      tmp2 = _mm_shuffle_epi32(tmp4, 0xaa);
      tmp4 = _mm_slli_si128(*tmp3, 0x4);
      *tmp3 = _mm_xor_si128(*tmp3, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp3 = _mm_xor_si128(*tmp3, tmp4);
      tmp4 = _mm_slli_si128(tmp4, 0x4);
      *tmp3 = _mm_xor_si128(*tmp3, tmp4);
      *tmp3 = _mm_xor_si128(*tmp3, tmp2);
    }

    void AES_256_Key_Expansion(const unsigned char *user_key, unsigned char *key) {
      __m128i tmp1, tmp2, tmp3;
      __m128i *key_sched = (__m128i *) key;
      tmp1 = _mm_loadu_si128((__m128i *) user_key);
      tmp3 = _mm_loadu_si128((__m128i *) (user_key + 16));
      key_sched[0] = tmp1;
      key_sched[1] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x01);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[2] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[3] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x02);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[4] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[5] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x04);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[6] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[7] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x08);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[8] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[9] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x10);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[10] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[11] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x20);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[12] = tmp1;
      KEY_256_ASSIST_2(&tmp1, &tmp3);
      key_sched[13] = tmp3;
      tmp2 = _mm_aeskeygenassist_si128(tmp3, 0x40);
      KEY_256_ASSIST_1(&tmp1, &tmp2);
      key_sched[14] = tmp1;
    }


  public:

    //the parameter key contains the key, and key bits can only be 128, 192 or 256 bits
    Aes(unsigned char key[key_bytes]) : round_keys() {
      constexpr bool valid_aes_key_bit_size = key_bits == 128 || key_bits == 192 || key_bits == 256;
      static_assert(valid_aes_key_bit_size, "The valid values are only: 128, 192 & 256");

      if constexpr (key_bits == 128) {
        AES_128_Key_Expansion(key, round_keys);
      } else if constexpr (key_bits == 192) {
        AES_192_Key_Expansion(key, round_keys);
      } else if constexpr (key_bits == 256) {
        AES_256_Key_Expansion(key, round_keys);
      }

    }

    const unsigned char* getRoundKeys() const { return round_keys; }

    ~Aes() {
      std::memset(round_keys, 0x00, round_keys_size);
    }

    //aes encryption of a 16 bytes block. it will overwrite the input data
    void encrypt_block(unsigned char *block) {
      // load the current block & current round key into the registers
      __m128i *xmm_round_keys = (__m128i *) round_keys;
      __m128i state = _mm_loadu_si128((__m128i *) &block[0]);

      // original key
      state = _mm_xor_si128(state, xmm_round_keys[0]);

      // perform usual rounds
      for (size_t i = 1; i < Nr - 1; i += 2) {
        state = _mm_aesenc_si128(state, xmm_round_keys[i]);
        state = _mm_aesenc_si128(state, xmm_round_keys[i + 1]);
      }

      // last round
      state = _mm_aesenc_si128(state, xmm_round_keys[Nr - 1]);
      state = _mm_aesenclast_si128(state, xmm_round_keys[Nr]);

      // store from register to array
      _mm_storeu_si128((__m128i *) (block), state);
    }

    //aes decryption of a 16 bytes block. it will overwrite the input data
    void decrypt_block(unsigned char *block) {
      // load the current block & current round key into the registers
      __m128i *xmm_round_keys = (__m128i *) round_keys;
      __m128i state = _mm_loadu_si128((__m128i *) &block[0]);

      // first round
      state = _mm_xor_si128(state, xmm_round_keys[Nr]);

      // usual rounds
      for (size_t i = Nr - 1; i > 1; i -= 2) {
        state = _mm_aesdec_si128(state, _mm_aesimc_si128(xmm_round_keys[i]));
        state = _mm_aesdec_si128(state, _mm_aesimc_si128(xmm_round_keys[i - 1]));
      }

      // last round
      state = _mm_aesdec_si128(state, _mm_aesimc_si128(xmm_round_keys[1]));
      state = _mm_aesdeclast_si128(state, xmm_round_keys[0]);

      // store from register to array
      _mm_storeu_si128((__m128i *) block, state);
    }
  };
}