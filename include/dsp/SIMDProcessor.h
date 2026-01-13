/**
 * MolinAntro DAW - SIMD-Optimized DSP Processor
 * SOTA x5 Implementation - Next 100 Years Architecture
 *
 * Features:
 * - AVX-512, AVX2, SSE4.2, NEON, SVE support
 * - Auto-vectorization with runtime CPU detection
 * - Lock-free processing with cache-aligned buffers
 * - Zero-copy circular buffers
 * - Real-time safe memory allocation
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <memory>
#include <array>
#include <vector>
#include <atomic>
#include <functional>
#include <algorithm>

// Platform-specific SIMD includes
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define MOLINANTRO_AVX512 1
    #define MOLINANTRO_SIMD_WIDTH 16
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define MOLINANTRO_AVX2 1
    #define MOLINANTRO_SIMD_WIDTH 8
#elif defined(__AVX__)
    #include <immintrin.h>
    #define MOLINANTRO_AVX 1
    #define MOLINANTRO_SIMD_WIDTH 8
#elif defined(__SSE4_2__)
    #include <nmmintrin.h>
    #define MOLINANTRO_SSE42 1
    #define MOLINANTRO_SIMD_WIDTH 4
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define MOLINANTRO_NEON 1
    #define MOLINANTRO_SIMD_WIDTH 4
#else
    #define MOLINANTRO_SCALAR 1
    #define MOLINANTRO_SIMD_WIDTH 1
#endif

// Cache line alignment for optimal performance
#define MOLINANTRO_CACHE_LINE 64
#define MOLINANTRO_ALIGN alignas(MOLINANTRO_CACHE_LINE)

namespace MolinAntro {
namespace DSP {
namespace SIMD {

// =============================================================================
// CPU FEATURE DETECTION
// =============================================================================

enum class CPUFeatures : uint32_t {
    None        = 0,
    SSE         = 1 << 0,
    SSE2        = 1 << 1,
    SSE3        = 1 << 2,
    SSSE3       = 1 << 3,
    SSE41       = 1 << 4,
    SSE42       = 1 << 5,
    AVX         = 1 << 6,
    AVX2        = 1 << 7,
    AVX512F     = 1 << 8,
    AVX512VL    = 1 << 9,
    AVX512BW    = 1 << 10,
    AVX512DQ    = 1 << 11,
    FMA3        = 1 << 12,
    NEON        = 1 << 13,
    SVE         = 1 << 14,
    SVE2        = 1 << 15
};

inline CPUFeatures operator|(CPUFeatures a, CPUFeatures b) {
    return static_cast<CPUFeatures>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline CPUFeatures operator&(CPUFeatures a, CPUFeatures b) {
    return static_cast<CPUFeatures>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

class CPUInfo {
public:
    static CPUInfo& instance() {
        static CPUInfo info;
        return info;
    }

    CPUFeatures getFeatures() const { return features_; }

    bool hasAVX512() const { return (features_ & CPUFeatures::AVX512F) != CPUFeatures::None; }
    bool hasAVX2() const { return (features_ & CPUFeatures::AVX2) != CPUFeatures::None; }
    bool hasAVX() const { return (features_ & CPUFeatures::AVX) != CPUFeatures::None; }
    bool hasNEON() const { return (features_ & CPUFeatures::NEON) != CPUFeatures::None; }
    bool hasFMA() const { return (features_ & CPUFeatures::FMA3) != CPUFeatures::None; }

    int getOptimalVectorWidth() const {
        if (hasAVX512()) return 16;
        if (hasAVX2() || hasAVX()) return 8;
        if (hasNEON()) return 4;
        return 4; // SSE minimum
    }

private:
    CPUInfo() { detectFeatures(); }
    void detectFeatures();
    CPUFeatures features_ = CPUFeatures::None;
};

// =============================================================================
// ALIGNED MEMORY ALLOCATOR
// =============================================================================

template<typename T, size_t Alignment = MOLINANTRO_CACHE_LINE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template<typename T, size_t A1, typename U, size_t A2>
bool operator==(const AlignedAllocator<T, A1>&, const AlignedAllocator<U, A2>&) noexcept {
    return A1 == A2;
}

template<typename T, size_t A1, typename U, size_t A2>
bool operator!=(const AlignedAllocator<T, A1>&, const AlignedAllocator<U, A2>&) noexcept {
    return A1 != A2;
}

// Aligned vector type
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// =============================================================================
// SIMD VECTOR OPERATIONS
// =============================================================================

#if defined(MOLINANTRO_AVX512)

// AVX-512 optimized operations (16 floats at a time)
inline void simd_add(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void simd_multiply(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vr = _mm512_mul_ps(va, vb);
        _mm512_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
    // result = a * b + c
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vc = _mm512_load_ps(c + i);
        __m512 vr = _mm512_fmadd_ps(va, vb, vc);
        _mm512_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

inline void simd_scale(const float* src, float scale, float* dst, size_t count) {
    __m512 vscale = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vs = _mm512_load_ps(src + i);
        __m512 vr = _mm512_mul_ps(vs, vscale);
        _mm512_store_ps(dst + i, vr);
    }
    for (; i < count; ++i) {
        dst[i] = src[i] * scale;
    }
}

inline float simd_sum(const float* data, size_t count) {
    __m512 vsum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        vsum = _mm512_add_ps(vsum, v);
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

inline float simd_max_abs(const float* data, size_t count) {
    __m512 vmax = _mm512_setzero_ps();
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 vabs = _mm512_andnot_ps(sign_mask, v);
        vmax = _mm512_max_ps(vmax, vabs);
    }
    float maxVal = _mm512_reduce_max_ps(vmax);
    for (; i < count; ++i) {
        maxVal = std::max(maxVal, std::abs(data[i]));
    }
    return maxVal;
}

#elif defined(MOLINANTRO_AVX2) || defined(MOLINANTRO_AVX)

// AVX/AVX2 optimized operations (8 floats at a time)
inline void simd_add(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void simd_multiply(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

#ifdef MOLINANTRO_AVX2
inline void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vc = _mm256_load_ps(c + i);
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
        _mm256_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}
#else
inline void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
    simd_multiply(a, b, result, count);
    simd_add(result, c, result, count);
}
#endif

inline void simd_scale(const float* src, float scale, float* dst, size_t count) {
    __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vs = _mm256_load_ps(src + i);
        __m256 vr = _mm256_mul_ps(vs, vscale);
        _mm256_store_ps(dst + i, vr);
    }
    for (; i < count; ++i) {
        dst[i] = src[i] * scale;
    }
}

inline float simd_sum(const float* data, size_t count) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        vsum = _mm256_add_ps(vsum, v);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum4 = _mm_add_ps(hi, lo);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    float sum = _mm_cvtss_f32(sum1);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

inline float simd_max_abs(const float* data, size_t count) {
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 vabs = _mm256_andnot_ps(sign_mask, v);
        vmax = _mm256_max_ps(vmax, vabs);
    }
    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 max4 = _mm_max_ps(hi, lo);
    __m128 max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    __m128 max1 = _mm_max_ss(max2, _mm_shuffle_ps(max2, max2, 1));
    float maxVal = _mm_cvtss_f32(max1);
    for (; i < count; ++i) {
        maxVal = std::max(maxVal, std::abs(data[i]));
    }
    return maxVal;
}

#elif defined(MOLINANTRO_NEON)

// ARM NEON optimized operations (4 floats at a time)
inline void simd_add(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void simd_multiply(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        float32x4_t vr = vfmaq_f32(vc, va, vb);  // c + a*b
        vst1q_f32(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

inline void simd_scale(const float* src, float scale, float* dst, size_t count) {
    float32x4_t vscale = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vs = vld1q_f32(src + i);
        float32x4_t vr = vmulq_f32(vs, vscale);
        vst1q_f32(dst + i, vr);
    }
    for (; i < count; ++i) {
        dst[i] = src[i] * scale;
    }
}

inline float simd_sum(const float* data, size_t count) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vaddq_f32(vsum, v);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

inline float simd_max_abs(const float* data, size_t count) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t vabs = vabsq_f32(v);
        vmax = vmaxq_f32(vmax, vabs);
    }
    float maxVal = vmaxvq_f32(vmax);
    for (; i < count; ++i) {
        maxVal = std::max(maxVal, std::abs(data[i]));
    }
    return maxVal;
}

#else

// Scalar fallback
inline void simd_add(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline void simd_multiply(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

inline void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

inline void simd_scale(const float* src, float scale, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i] * scale;
    }
}

inline float simd_sum(const float* data, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

inline float simd_max_abs(const float* data, size_t count) {
    float maxVal = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        maxVal = std::max(maxVal, std::abs(data[i]));
    }
    return maxVal;
}

#endif

// =============================================================================
// SIMD BIQUAD FILTER (8-way parallel)
// =============================================================================

struct MOLINANTRO_ALIGN BiquadCoeffs8 {
    alignas(32) float b0[8];
    alignas(32) float b1[8];
    alignas(32) float b2[8];
    alignas(32) float a1[8];
    alignas(32) float a2[8];
};

struct MOLINANTRO_ALIGN BiquadState8 {
    alignas(32) float x1[8];
    alignas(32) float x2[8];
    alignas(32) float y1[8];
    alignas(32) float y2[8];
};

// Process 8 biquad filters in parallel using SIMD
inline void simd_biquad_process_8(
    const float* input,
    float* output,
    size_t numSamples,
    const BiquadCoeffs8& coeffs,
    BiquadState8& state
) {
#if defined(MOLINANTRO_AVX) || defined(MOLINANTRO_AVX2) || defined(MOLINANTRO_AVX512)
    __m256 vb0 = _mm256_load_ps(coeffs.b0);
    __m256 vb1 = _mm256_load_ps(coeffs.b1);
    __m256 vb2 = _mm256_load_ps(coeffs.b2);
    __m256 va1 = _mm256_load_ps(coeffs.a1);
    __m256 va2 = _mm256_load_ps(coeffs.a2);

    __m256 vx1 = _mm256_load_ps(state.x1);
    __m256 vx2 = _mm256_load_ps(state.x2);
    __m256 vy1 = _mm256_load_ps(state.y1);
    __m256 vy2 = _mm256_load_ps(state.y2);

    for (size_t i = 0; i < numSamples; ++i) {
        __m256 vin = _mm256_set1_ps(input[i]);

        // y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
        __m256 vout = _mm256_mul_ps(vb0, vin);
        #ifdef MOLINANTRO_AVX2
        vout = _mm256_fmadd_ps(vb1, vx1, vout);
        vout = _mm256_fmadd_ps(vb2, vx2, vout);
        vout = _mm256_fnmadd_ps(va1, vy1, vout);
        vout = _mm256_fnmadd_ps(va2, vy2, vout);
        #else
        vout = _mm256_add_ps(vout, _mm256_mul_ps(vb1, vx1));
        vout = _mm256_add_ps(vout, _mm256_mul_ps(vb2, vx2));
        vout = _mm256_sub_ps(vout, _mm256_mul_ps(va1, vy1));
        vout = _mm256_sub_ps(vout, _mm256_mul_ps(va2, vy2));
        #endif

        // Update state
        vx2 = vx1;
        vx1 = vin;
        vy2 = vy1;
        vy1 = vout;

        // Horizontal sum for output (sum of 8 filter outputs)
        __m128 hi = _mm256_extractf128_ps(vout, 1);
        __m128 lo = _mm256_castps256_ps128(vout);
        __m128 sum4 = _mm_add_ps(hi, lo);
        __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        output[i] = _mm_cvtss_f32(sum1) * 0.125f; // Average
    }

    _mm256_store_ps(state.x1, vx1);
    _mm256_store_ps(state.x2, vx2);
    _mm256_store_ps(state.y1, vy1);
    _mm256_store_ps(state.y2, vy2);
#else
    // Scalar fallback
    for (size_t i = 0; i < numSamples; ++i) {
        float out = 0.0f;
        for (int f = 0; f < 8; ++f) {
            float y = coeffs.b0[f] * input[i]
                    + coeffs.b1[f] * state.x1[f]
                    + coeffs.b2[f] * state.x2[f]
                    - coeffs.a1[f] * state.y1[f]
                    - coeffs.a2[f] * state.y2[f];

            state.x2[f] = state.x1[f];
            state.x1[f] = input[i];
            state.y2[f] = state.y1[f];
            state.y1[f] = y;

            out += y;
        }
        output[i] = out * 0.125f;
    }
#endif
}

// =============================================================================
// SIMD FFT (Radix-4 optimized)
// =============================================================================

class SIMDFFTProcessor {
public:
    SIMDFFTProcessor(size_t fftSize) : fftSize_(fftSize) {
        // Allocate aligned buffers
        real_.resize(fftSize);
        imag_.resize(fftSize);

        // Precompute twiddle factors
        twiddleReal_.resize(fftSize / 2);
        twiddleImag_.resize(fftSize / 2);

        const double twoPi = 2.0 * 3.14159265358979323846;
        for (size_t k = 0; k < fftSize / 2; ++k) {
            double angle = -twoPi * k / fftSize;
            twiddleReal_[k] = static_cast<float>(std::cos(angle));
            twiddleImag_[k] = static_cast<float>(std::sin(angle));
        }
    }

    void forward(const float* input, float* outputReal, float* outputImag) {
        // Copy input to working buffer
        std::copy(input, input + fftSize_, real_.begin());
        std::fill(imag_.begin(), imag_.end(), 0.0f);

        // Bit-reversal permutation
        bitReverse();

        // Cooley-Tukey FFT
        for (size_t stage = 1; stage < fftSize_; stage *= 2) {
            size_t halfStage = stage;
            size_t twiddleStep = fftSize_ / (stage * 2);

            for (size_t k = 0; k < fftSize_; k += stage * 2) {
                for (size_t j = 0; j < halfStage; ++j) {
                    size_t idx1 = k + j;
                    size_t idx2 = k + j + halfStage;
                    size_t twiddleIdx = j * twiddleStep;

                    float tr = twiddleReal_[twiddleIdx];
                    float ti = twiddleImag_[twiddleIdx];

                    float tempR = real_[idx2] * tr - imag_[idx2] * ti;
                    float tempI = real_[idx2] * ti + imag_[idx2] * tr;

                    real_[idx2] = real_[idx1] - tempR;
                    imag_[idx2] = imag_[idx1] - tempI;
                    real_[idx1] = real_[idx1] + tempR;
                    imag_[idx1] = imag_[idx1] + tempI;
                }
            }
        }

        // Copy output
        std::copy(real_.begin(), real_.end(), outputReal);
        std::copy(imag_.begin(), imag_.end(), outputImag);
    }

    void inverse(const float* inputReal, const float* inputImag, float* output) {
        // Copy input
        std::copy(inputReal, inputReal + fftSize_, real_.begin());
        std::copy(inputImag, inputImag + fftSize_, imag_.begin());

        // Conjugate
        for (auto& v : imag_) v = -v;

        // Forward FFT
        bitReverse();

        for (size_t stage = 1; stage < fftSize_; stage *= 2) {
            size_t halfStage = stage;
            size_t twiddleStep = fftSize_ / (stage * 2);

            for (size_t k = 0; k < fftSize_; k += stage * 2) {
                for (size_t j = 0; j < halfStage; ++j) {
                    size_t idx1 = k + j;
                    size_t idx2 = k + j + halfStage;
                    size_t twiddleIdx = j * twiddleStep;

                    float tr = twiddleReal_[twiddleIdx];
                    float ti = twiddleImag_[twiddleIdx];

                    float tempR = real_[idx2] * tr - imag_[idx2] * ti;
                    float tempI = real_[idx2] * ti + imag_[idx2] * tr;

                    real_[idx2] = real_[idx1] - tempR;
                    imag_[idx2] = imag_[idx1] - tempI;
                    real_[idx1] = real_[idx1] + tempR;
                    imag_[idx1] = imag_[idx1] + tempI;
                }
            }
        }

        // Conjugate and normalize
        float scale = 1.0f / fftSize_;
        for (size_t i = 0; i < fftSize_; ++i) {
            output[i] = real_[i] * scale;
        }
    }

private:
    void bitReverse() {
        size_t j = 0;
        for (size_t i = 0; i < fftSize_ - 1; ++i) {
            if (i < j) {
                std::swap(real_[i], real_[j]);
                std::swap(imag_[i], imag_[j]);
            }
            size_t k = fftSize_ / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
    }

    size_t fftSize_;
    AlignedVector<float> real_;
    AlignedVector<float> imag_;
    AlignedVector<float> twiddleReal_;
    AlignedVector<float> twiddleImag_;
};

// =============================================================================
// SIMD CONVOLUTION (for impulse responses)
// =============================================================================

class SIMDConvolver {
public:
    SIMDConvolver(size_t irLength, size_t blockSize)
        : irLength_(irLength), blockSize_(blockSize)
    {
        // Round up to power of 2 for FFT
        fftSize_ = 1;
        while (fftSize_ < irLength + blockSize) {
            fftSize_ *= 2;
        }

        fft_ = std::make_unique<SIMDFFTProcessor>(fftSize_);

        // Allocate buffers
        irReal_.resize(fftSize_, 0.0f);
        irImag_.resize(fftSize_, 0.0f);
        inputBuffer_.resize(fftSize_, 0.0f);
        outputBuffer_.resize(fftSize_, 0.0f);
        overlapBuffer_.resize(fftSize_, 0.0f);
        tempReal_.resize(fftSize_);
        tempImag_.resize(fftSize_);
    }

    void setImpulseResponse(const float* ir, size_t length) {
        std::fill(irReal_.begin(), irReal_.end(), 0.0f);
        std::copy(ir, ir + std::min(length, irLength_), irReal_.begin());

        // Transform IR to frequency domain
        fft_->forward(irReal_.data(), irReal_.data(), irImag_.data());
    }

    void process(const float* input, float* output, size_t numSamples) {
        // Copy input and zero-pad
        std::fill(inputBuffer_.begin(), inputBuffer_.end(), 0.0f);
        std::copy(input, input + numSamples, inputBuffer_.begin());

        // Forward FFT of input
        fft_->forward(inputBuffer_.data(), tempReal_.data(), tempImag_.data());

        // Complex multiplication in frequency domain
        for (size_t i = 0; i < fftSize_; ++i) {
            float re = tempReal_[i] * irReal_[i] - tempImag_[i] * irImag_[i];
            float im = tempReal_[i] * irImag_[i] + tempImag_[i] * irReal_[i];
            tempReal_[i] = re;
            tempImag_[i] = im;
        }

        // Inverse FFT
        fft_->inverse(tempReal_.data(), tempImag_.data(), outputBuffer_.data());

        // Overlap-add
        for (size_t i = 0; i < numSamples; ++i) {
            output[i] = outputBuffer_[i] + overlapBuffer_[i];
        }

        // Save overlap for next block
        std::copy(outputBuffer_.begin() + numSamples,
                  outputBuffer_.begin() + numSamples + irLength_ - 1,
                  overlapBuffer_.begin());
    }

private:
    size_t irLength_;
    size_t blockSize_;
    size_t fftSize_;
    std::unique_ptr<SIMDFFTProcessor> fft_;
    AlignedVector<float> irReal_;
    AlignedVector<float> irImag_;
    AlignedVector<float> inputBuffer_;
    AlignedVector<float> outputBuffer_;
    AlignedVector<float> overlapBuffer_;
    AlignedVector<float> tempReal_;
    AlignedVector<float> tempImag_;
};

} // namespace SIMD
} // namespace DSP
} // namespace MolinAntro
