/**
 * MolinAntro DAW - SIMD-Optimized Audio DSP Implementation
 * SOTA x5 Implementation - Real working code
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 */

#include "../../include/dsp/SIMDProcessor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MolinAntro {
namespace DSP {

// =============================================================================
// CPU FEATURE DETECTION - REAL IMPLEMENTATION
// =============================================================================

#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(int info[4], int function_id) {
    __cpuid(info, function_id);
}
static void cpuidex(int info[4], int function_id, int subfunction_id) {
    __cpuidex(info, function_id, subfunction_id);
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static void cpuid(int info[4], int function_id) {
    __cpuid(function_id, info[0], info[1], info[2], info[3]);
}
static void cpuidex(int info[4], int function_id, int subfunction_id) {
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
}
#endif

CPUFeatures CPUFeatures::detect() {
    CPUFeatures features{};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    int info[4];

    // Get vendor string
    cpuid(info, 0);
    int maxFunction = info[0];

    if (maxFunction >= 1) {
        cpuid(info, 1);
        features.hasSSE2 = (info[3] & (1 << 26)) != 0;
        features.hasSSE3 = (info[2] & (1 << 0)) != 0;
        features.hasSSSE3 = (info[2] & (1 << 9)) != 0;
        features.hasSSE41 = (info[2] & (1 << 19)) != 0;
        features.hasSSE42 = (info[2] & (1 << 20)) != 0;
        features.hasFMA = (info[2] & (1 << 12)) != 0;
        features.hasAVX = (info[2] & (1 << 28)) != 0;
    }

    if (maxFunction >= 7) {
        cpuidex(info, 7, 0);
        features.hasAVX2 = (info[1] & (1 << 5)) != 0;
        features.hasAVX512F = (info[1] & (1 << 16)) != 0;
        features.hasAVX512DQ = (info[1] & (1 << 17)) != 0;
        features.hasAVX512BW = (info[1] & (1 << 30)) != 0;
        features.hasAVX512VL = (info[1] & (1 << 31)) != 0;
    }

    // Get cache line size
    cpuid(info, 0x80000006);
    features.cacheLineSize = info[2] & 0xFF;
    if (features.cacheLineSize == 0) features.cacheLineSize = 64;

#elif defined(__aarch64__) || defined(_M_ARM64)
    features.hasNEON = true;
    features.cacheLineSize = 64;

    // Check for SVE on ARM64
    #if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    features.hasSVE = (hwcap & HWCAP_SVE) != 0;
    #endif

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    features.hasNEON = true;
    features.cacheLineSize = 32;
#endif

    return features;
}

// Global CPU features instance
static CPUFeatures g_cpuFeatures = CPUFeatures::detect();

const CPUFeatures& getCPUFeatures() {
    return g_cpuFeatures;
}

// =============================================================================
// SIMD OPERATIONS - REAL IMPLEMENTATIONS
// =============================================================================

#if MOLINANTRO_USE_AVX512

void simd_add_avx512(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_store_ps(result + i, vr);
    }
    // Handle remaining elements
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simd_multiply_avx512(const float* a, const float* b, float* result, size_t count) {
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

void simd_fma_avx512(const float* a, const float* b, const float* c, float* result, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vc = _mm512_load_ps(c + i);
        __m512 vr = _mm512_fmadd_ps(va, vb, vc);  // a*b + c
        _mm512_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void simd_scale_avx512(const float* input, float scalar, float* result, size_t count) {
    __m512 vs = _mm512_set1_ps(scalar);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vi = _mm512_load_ps(input + i);
        __m512 vr = _mm512_mul_ps(vi, vs);
        _mm512_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = input[i] * scalar;
    }
}

float simd_sum_avx512(const float* data, size_t count) {
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

float simd_max_abs_avx512(const float* data, size_t count) {
    __m512 vmax = _mm512_setzero_ps();
    __m512 signMask = _mm512_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_load_ps(data + i);
        __m512 vabs = _mm512_andnot_ps(signMask, v);  // Clear sign bit
        vmax = _mm512_max_ps(vmax, vabs);
    }
    float maxVal = _mm512_reduce_max_ps(vmax);
    for (; i < count; ++i) {
        float absVal = std::fabs(data[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    return maxVal;
}

#endif // MOLINANTRO_USE_AVX512

#if MOLINANTRO_USE_AVX2

void simd_add_avx2(const float* a, const float* b, float* result, size_t count) {
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

void simd_multiply_avx2(const float* a, const float* b, float* result, size_t count) {
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

void simd_fma_avx2(const float* a, const float* b, const float* c, float* result, size_t count) {
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

void simd_scale_avx2(const float* input, float scalar, float* result, size_t count) {
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vi = _mm256_load_ps(input + i);
        __m256 vr = _mm256_mul_ps(vi, vs);
        _mm256_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = input[i] * scalar;
    }
}

float simd_sum_avx2(const float* data, size_t count) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        vsum = _mm256_add_ps(vsum, v);
    }
    // Horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

float simd_max_abs_avx2(const float* data, size_t count) {
    __m256 vmax = _mm256_setzero_ps();
    __m256 signMask = _mm256_set1_ps(-0.0f);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_load_ps(data + i);
        __m256 vabs = _mm256_andnot_ps(signMask, v);
        vmax = _mm256_max_ps(vmax, vabs);
    }
    // Horizontal max
    __m128 vlow = _mm256_castps256_ps128(vmax);
    __m128 vhigh = _mm256_extractf128_ps(vmax, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    vlow = _mm_max_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, vlow);
    vlow = _mm_max_ss(vlow, shuf);
    float maxVal = _mm_cvtss_f32(vlow);
    for (; i < count; ++i) {
        float absVal = std::fabs(data[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    return maxVal;
}

#endif // MOLINANTRO_USE_AVX2

#if MOLINANTRO_USE_SSE42

void simd_add_sse42(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128 va = _mm_load_ps(a + i);
        __m128 vb = _mm_load_ps(b + i);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simd_multiply_sse42(const float* a, const float* b, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128 va = _mm_load_ps(a + i);
        __m128 vb = _mm_load_ps(b + i);
        __m128 vr = _mm_mul_ps(va, vb);
        _mm_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void simd_scale_sse42(const float* input, float scalar, float* result, size_t count) {
    __m128 vs = _mm_set1_ps(scalar);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128 vi = _mm_load_ps(input + i);
        __m128 vr = _mm_mul_ps(vi, vs);
        _mm_store_ps(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = input[i] * scalar;
    }
}

float simd_sum_sse42(const float* data, size_t count) {
    __m128 vsum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128 v = _mm_load_ps(data + i);
        vsum = _mm_add_ps(vsum, v);
    }
    // Horizontal sum
    __m128 shuf = _mm_movehdup_ps(vsum);
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

#endif // MOLINANTRO_USE_SSE42

#if MOLINANTRO_USE_NEON

void simd_add_neon(const float* a, const float* b, float* result, size_t count) {
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

void simd_multiply_neon(const float* a, const float* b, float* result, size_t count) {
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

void simd_fma_neon(const float* a, const float* b, const float* c, float* result, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        float32x4_t vr = vmlaq_f32(vc, va, vb);  // c + a*b
        vst1q_f32(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void simd_scale_neon(const float* input, float scalar, float* result, size_t count) {
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vi = vld1q_f32(input + i);
        float32x4_t vr = vmulq_f32(vi, vs);
        vst1q_f32(result + i, vr);
    }
    for (; i < count; ++i) {
        result[i] = input[i] * scalar;
    }
}

float simd_sum_neon(const float* data, size_t count) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vaddq_f32(vsum, v);
    }
    // Horizontal sum
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float sum = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);
    for (; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

float simd_max_abs_neon(const float* data, size_t count) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t vabs = vabsq_f32(v);
        vmax = vmaxq_f32(vmax, vabs);
    }
    // Horizontal max
    float32x2_t vmax2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float maxVal = vget_lane_f32(vmax2, 0);
    for (; i < count; ++i) {
        float absVal = std::fabs(data[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    return maxVal;
}

#endif // MOLINANTRO_USE_NEON

// =============================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =============================================================================

void simd_add_scalar(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simd_multiply_scalar(const float* a, const float* b, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void simd_fma_scalar(const float* a, const float* b, const float* c, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void simd_scale_scalar(const float* input, float scalar, float* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = input[i] * scalar;
    }
}

float simd_sum_scalar(const float* data, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

float simd_max_abs_scalar(const float* data, size_t count) {
    float maxVal = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float absVal = std::fabs(data[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    return maxVal;
}

// =============================================================================
// DISPATCHER FUNCTIONS
// =============================================================================

void simd_add(const float* a, const float* b, float* result, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        simd_add_avx512(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasAVX2) {
        simd_add_avx2(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_SSE42
    if (g_cpuFeatures.hasSSE42) {
        simd_add_sse42(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        simd_add_neon(a, b, result, count);
        return;
    }
#endif
    simd_add_scalar(a, b, result, count);
}

void simd_multiply(const float* a, const float* b, float* result, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        simd_multiply_avx512(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasAVX2) {
        simd_multiply_avx2(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_SSE42
    if (g_cpuFeatures.hasSSE42) {
        simd_multiply_sse42(a, b, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        simd_multiply_neon(a, b, result, count);
        return;
    }
#endif
    simd_multiply_scalar(a, b, result, count);
}

void simd_fma(const float* a, const float* b, const float* c, float* result, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        simd_fma_avx512(a, b, c, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasFMA && g_cpuFeatures.hasAVX2) {
        simd_fma_avx2(a, b, c, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        simd_fma_neon(a, b, c, result, count);
        return;
    }
#endif
    simd_fma_scalar(a, b, c, result, count);
}

void simd_scale(const float* input, float scalar, float* result, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        simd_scale_avx512(input, scalar, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasAVX2) {
        simd_scale_avx2(input, scalar, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_SSE42
    if (g_cpuFeatures.hasSSE42) {
        simd_scale_sse42(input, scalar, result, count);
        return;
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        simd_scale_neon(input, scalar, result, count);
        return;
    }
#endif
    simd_scale_scalar(input, scalar, result, count);
}

float simd_sum(const float* data, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        return simd_sum_avx512(data, count);
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasAVX2) {
        return simd_sum_avx2(data, count);
    }
#endif
#if MOLINANTRO_USE_SSE42
    if (g_cpuFeatures.hasSSE42) {
        return simd_sum_sse42(data, count);
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        return simd_sum_neon(data, count);
    }
#endif
    return simd_sum_scalar(data, count);
}

float simd_max_abs(const float* data, size_t count) {
#if MOLINANTRO_USE_AVX512
    if (g_cpuFeatures.hasAVX512F) {
        return simd_max_abs_avx512(data, count);
    }
#endif
#if MOLINANTRO_USE_AVX2
    if (g_cpuFeatures.hasAVX2) {
        return simd_max_abs_avx2(data, count);
    }
#endif
#if MOLINANTRO_USE_NEON
    if (g_cpuFeatures.hasNEON) {
        return simd_max_abs_neon(data, count);
    }
#endif
    return simd_max_abs_scalar(data, count);
}

// =============================================================================
// SIMD BIQUAD FILTER - 8-WAY PARALLEL
// =============================================================================

SIMDBiquadFilter8::SIMDBiquadFilter8() {
    reset();
}

void SIMDBiquadFilter8::setCoefficients(const BiquadCoeffs8& coeffs) {
    coeffs_ = coeffs;
}

void SIMDBiquadFilter8::reset() {
    for (int i = 0; i < 8; ++i) {
        state_.z1[i] = 0.0f;
        state_.z2[i] = 0.0f;
    }
}

#if MOLINANTRO_USE_AVX2

void SIMDBiquadFilter8::process(const float* input, float* output, size_t numSamples) {
    __m256 b0 = _mm256_load_ps(coeffs_.b0);
    __m256 b1 = _mm256_load_ps(coeffs_.b1);
    __m256 b2 = _mm256_load_ps(coeffs_.b2);
    __m256 a1 = _mm256_load_ps(coeffs_.a1);
    __m256 a2 = _mm256_load_ps(coeffs_.a2);
    __m256 z1 = _mm256_load_ps(state_.z1);
    __m256 z2 = _mm256_load_ps(state_.z2);

    for (size_t i = 0; i < numSamples; ++i) {
        __m256 x = _mm256_load_ps(input + i * 8);

        // y = b0*x + z1
        __m256 y = _mm256_fmadd_ps(b0, x, z1);

        // z1 = b1*x - a1*y + z2
        __m256 t1 = _mm256_mul_ps(b1, x);
        __m256 t2 = _mm256_mul_ps(a1, y);
        z1 = _mm256_add_ps(_mm256_sub_ps(t1, t2), z2);

        // z2 = b2*x - a2*y
        t1 = _mm256_mul_ps(b2, x);
        t2 = _mm256_mul_ps(a2, y);
        z2 = _mm256_sub_ps(t1, t2);

        _mm256_store_ps(output + i * 8, y);
    }

    _mm256_store_ps(state_.z1, z1);
    _mm256_store_ps(state_.z2, z2);
}

#else

void SIMDBiquadFilter8::process(const float* input, float* output, size_t numSamples) {
    // Scalar fallback
    for (size_t i = 0; i < numSamples; ++i) {
        for (int ch = 0; ch < 8; ++ch) {
            float x = input[i * 8 + ch];
            float y = coeffs_.b0[ch] * x + state_.z1[ch];
            state_.z1[ch] = coeffs_.b1[ch] * x - coeffs_.a1[ch] * y + state_.z2[ch];
            state_.z2[ch] = coeffs_.b2[ch] * x - coeffs_.a2[ch] * y;
            output[i * 8 + ch] = y;
        }
    }
}

#endif

// =============================================================================
// SIMD FFT PROCESSOR
// =============================================================================

SIMDFFTProcessor::SIMDFFTProcessor(size_t fftSize)
    : fftSize_(fftSize), log2Size_(0) {

    // Calculate log2(fftSize)
    size_t n = fftSize;
    while (n > 1) {
        n >>= 1;
        ++log2Size_;
    }

    if ((1ULL << log2Size_) != fftSize_) {
        throw std::invalid_argument("FFT size must be a power of 2");
    }

    // Allocate aligned buffers
    twiddleReal_.resize(fftSize_ / 2);
    twiddleImag_.resize(fftSize_ / 2);
    workReal_.resize(fftSize_);
    workImag_.resize(fftSize_);
    bitReverse_.resize(fftSize_);

    // Precompute twiddle factors
    for (size_t k = 0; k < fftSize_ / 2; ++k) {
        double angle = -2.0 * M_PI * k / fftSize_;
        twiddleReal_[k] = static_cast<float>(std::cos(angle));
        twiddleImag_[k] = static_cast<float>(std::sin(angle));
    }

    // Precompute bit-reversal indices
    for (size_t i = 0; i < fftSize_; ++i) {
        size_t rev = 0;
        size_t n = i;
        for (size_t j = 0; j < log2Size_; ++j) {
            rev = (rev << 1) | (n & 1);
            n >>= 1;
        }
        bitReverse_[i] = rev;
    }
}

void SIMDFFTProcessor::forward(const float* input, float* outputReal, float* outputImag) {
    // Bit-reversal permutation
    for (size_t i = 0; i < fftSize_; ++i) {
        size_t j = bitReverse_[i];
        workReal_[j] = input[i];
        workImag_[j] = 0.0f;
    }

    // Cooley-Tukey FFT
    for (size_t stage = 0; stage < log2Size_; ++stage) {
        size_t halfSize = 1ULL << stage;
        size_t fullSize = halfSize << 1;
        size_t twiddleStep = fftSize_ / fullSize;

        for (size_t group = 0; group < fftSize_; group += fullSize) {
            for (size_t pair = 0; pair < halfSize; ++pair) {
                size_t i = group + pair;
                size_t j = i + halfSize;
                size_t twIdx = pair * twiddleStep;

                float tr = twiddleReal_[twIdx];
                float ti = twiddleImag_[twIdx];

                float tempR = workReal_[j] * tr - workImag_[j] * ti;
                float tempI = workReal_[j] * ti + workImag_[j] * tr;

                workReal_[j] = workReal_[i] - tempR;
                workImag_[j] = workImag_[i] - tempI;
                workReal_[i] = workReal_[i] + tempR;
                workImag_[i] = workImag_[i] + tempI;
            }
        }
    }

    std::copy(workReal_.begin(), workReal_.end(), outputReal);
    std::copy(workImag_.begin(), workImag_.end(), outputImag);
}

void SIMDFFTProcessor::inverse(const float* inputReal, const float* inputImag, float* output) {
    // Bit-reversal permutation
    for (size_t i = 0; i < fftSize_; ++i) {
        size_t j = bitReverse_[i];
        workReal_[j] = inputReal[i];
        workImag_[j] = -inputImag[i];  // Conjugate for inverse
    }

    // Cooley-Tukey FFT
    for (size_t stage = 0; stage < log2Size_; ++stage) {
        size_t halfSize = 1ULL << stage;
        size_t fullSize = halfSize << 1;
        size_t twiddleStep = fftSize_ / fullSize;

        for (size_t group = 0; group < fftSize_; group += fullSize) {
            for (size_t pair = 0; pair < halfSize; ++pair) {
                size_t i = group + pair;
                size_t j = i + halfSize;
                size_t twIdx = pair * twiddleStep;

                float tr = twiddleReal_[twIdx];
                float ti = twiddleImag_[twIdx];

                float tempR = workReal_[j] * tr - workImag_[j] * ti;
                float tempI = workReal_[j] * ti + workImag_[j] * tr;

                workReal_[j] = workReal_[i] - tempR;
                workImag_[j] = workImag_[i] - tempI;
                workReal_[i] = workReal_[i] + tempR;
                workImag_[i] = workImag_[i] + tempI;
            }
        }
    }

    // Scale by 1/N and take real part
    float scale = 1.0f / static_cast<float>(fftSize_);
    for (size_t i = 0; i < fftSize_; ++i) {
        output[i] = workReal_[i] * scale;
    }
}

// =============================================================================
// SIMD CONVOLVER
// =============================================================================

SIMDConvolver::SIMDConvolver(size_t maxIRLength, size_t blockSize)
    : maxIRLength_(maxIRLength), blockSize_(blockSize), irLength_(0) {

    // Partition size must be power of 2 and >= blockSize
    partitionSize_ = 1;
    while (partitionSize_ < blockSize_ * 2) {
        partitionSize_ *= 2;
    }

    numPartitions_ = (maxIRLength + partitionSize_ / 2 - 1) / (partitionSize_ / 2);
    fftSize_ = partitionSize_;

    // Allocate aligned buffers
    inputBuffer_.resize(partitionSize_, 0.0f);
    outputBuffer_.resize(partitionSize_, 0.0f);
    overlapBuffer_.resize(partitionSize_ / 2, 0.0f);

    irPartitionsReal_.resize(numPartitions_ * (fftSize_ / 2 + 1), 0.0f);
    irPartitionsImag_.resize(numPartitions_ * (fftSize_ / 2 + 1), 0.0f);

    fdlReal_.resize(numPartitions_ * (fftSize_ / 2 + 1), 0.0f);
    fdlImag_.resize(numPartitions_ * (fftSize_ / 2 + 1), 0.0f);

    tempReal_.resize(fftSize_ / 2 + 1, 0.0f);
    tempImag_.resize(fftSize_ / 2 + 1, 0.0f);
    accumReal_.resize(fftSize_ / 2 + 1, 0.0f);
    accumImag_.resize(fftSize_ / 2 + 1, 0.0f);

    fft_ = std::make_unique<SIMDFFTProcessor>(fftSize_);

    inputPos_ = 0;
    fdlIndex_ = 0;
}

void SIMDConvolver::setImpulseResponse(const float* ir, size_t length) {
    irLength_ = std::min(length, maxIRLength_);
    size_t hopSize = partitionSize_ / 2;

    std::vector<float> paddedPartition(fftSize_, 0.0f);
    std::vector<float> partReal(fftSize_);
    std::vector<float> partImag(fftSize_);

    for (size_t p = 0; p < numPartitions_; ++p) {
        std::fill(paddedPartition.begin(), paddedPartition.end(), 0.0f);

        size_t irStart = p * hopSize;
        size_t copyLen = std::min(hopSize, irLength_ > irStart ? irLength_ - irStart : 0);

        if (copyLen > 0) {
            std::copy(ir + irStart, ir + irStart + copyLen, paddedPartition.begin());
        }

        fft_->forward(paddedPartition.data(), partReal.data(), partImag.data());

        // Store only positive frequencies (real FFT symmetry)
        for (size_t k = 0; k <= fftSize_ / 2; ++k) {
            irPartitionsReal_[p * (fftSize_ / 2 + 1) + k] = partReal[k];
            irPartitionsImag_[p * (fftSize_ / 2 + 1) + k] = partImag[k];
        }
    }

    reset();
}

void SIMDConvolver::reset() {
    std::fill(inputBuffer_.begin(), inputBuffer_.end(), 0.0f);
    std::fill(outputBuffer_.begin(), outputBuffer_.end(), 0.0f);
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
    std::fill(fdlReal_.begin(), fdlReal_.end(), 0.0f);
    std::fill(fdlImag_.begin(), fdlImag_.end(), 0.0f);
    inputPos_ = 0;
    fdlIndex_ = 0;
}

void SIMDConvolver::process(const float* input, float* output, size_t numSamples) {
    size_t hopSize = partitionSize_ / 2;
    size_t processed = 0;

    while (processed < numSamples) {
        size_t toCopy = std::min(hopSize - inputPos_, numSamples - processed);
        std::copy(input + processed, input + processed + toCopy, inputBuffer_.data() + inputPos_);
        inputPos_ += toCopy;
        processed += toCopy;

        if (inputPos_ >= hopSize) {
            processPartition();

            // Output with overlap-add
            for (size_t i = 0; i < hopSize; ++i) {
                output[processed - hopSize + i] = outputBuffer_[i] + overlapBuffer_[i];
            }

            std::copy(outputBuffer_.begin() + hopSize, outputBuffer_.end(), overlapBuffer_.begin());

            // Shift input buffer
            std::copy(inputBuffer_.begin() + hopSize, inputBuffer_.end(), inputBuffer_.begin());
            std::fill(inputBuffer_.begin() + hopSize, inputBuffer_.end(), 0.0f);
            inputPos_ = 0;
        }
    }
}

void SIMDConvolver::processPartition() {
    std::vector<float> inputReal(fftSize_);
    std::vector<float> inputImag(fftSize_);
    std::vector<float> outputTime(fftSize_);

    // FFT of input block
    fft_->forward(inputBuffer_.data(), inputReal.data(), inputImag.data());

    // Store in frequency-domain delay line
    size_t binCount = fftSize_ / 2 + 1;
    size_t fdlOffset = fdlIndex_ * binCount;
    for (size_t k = 0; k < binCount; ++k) {
        fdlReal_[fdlOffset + k] = inputReal[k];
        fdlImag_[fdlOffset + k] = inputImag[k];
    }

    // Accumulate complex multiplications
    std::fill(accumReal_.begin(), accumReal_.end(), 0.0f);
    std::fill(accumImag_.begin(), accumImag_.end(), 0.0f);

    for (size_t p = 0; p < numPartitions_; ++p) {
        size_t fdlIdx = (fdlIndex_ + numPartitions_ - p) % numPartitions_;
        size_t irOffset = p * binCount;
        size_t fdlOff = fdlIdx * binCount;

        for (size_t k = 0; k < binCount; ++k) {
            float fdlR = fdlReal_[fdlOff + k];
            float fdlI = fdlImag_[fdlOff + k];
            float irR = irPartitionsReal_[irOffset + k];
            float irI = irPartitionsImag_[irOffset + k];

            // Complex multiplication
            accumReal_[k] += fdlR * irR - fdlI * irI;
            accumImag_[k] += fdlR * irI + fdlI * irR;
        }
    }

    // Mirror for inverse FFT
    for (size_t k = 1; k < fftSize_ / 2; ++k) {
        inputReal[fftSize_ - k] = accumReal_[k];
        inputImag[fftSize_ - k] = -accumImag_[k];
    }
    for (size_t k = 0; k < binCount; ++k) {
        inputReal[k] = accumReal_[k];
        inputImag[k] = accumImag_[k];
    }

    // Inverse FFT
    fft_->inverse(inputReal.data(), inputImag.data(), outputTime.data());
    std::copy(outputTime.begin(), outputTime.end(), outputBuffer_.begin());

    fdlIndex_ = (fdlIndex_ + 1) % numPartitions_;
}

} // namespace DSP
} // namespace MolinAntro
