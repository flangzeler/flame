/*
flame.h by FLANGZELER

DATE: 05-02-2026 | RELEASE: 1.1.0

LICENSE: This code is released under the MIT License. See LICENSE file for details.

flame (FLANGZELER's MATH ENGINE) is a SIMD-based math library providing high-performance
mathematical operations for real-time rendering, game engines, and scientific computing.
Leverages SSE/AVX intrinsics with strict alignment guarantees and zero heap allocation.

*/

#pragma once

#include <immintrin.h>
#include <concepts>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cassert>
#include <memory>
#include <cstring>   // memcpy

#if defined(__GNUC__) || defined(__clang__)
#  define RESTRICT      __restrict__
#  define FORCE_INLINE  __attribute__((always_inline)) inline
#  define FLAME_PREFETCH_L1(p)  __builtin_prefetch((p), 0, 3)
#  define FLAME_PREFETCH_L2(p)  __builtin_prefetch((p), 0, 2)
#  define FLAME_PREFETCH_NT(p)  __builtin_prefetch((p), 0, 0)
#elif defined(_MSC_VER)
#  define RESTRICT      __restrict
#  define FORCE_INLINE  __forceinline
#  include <intrin.h>
#  define FLAME_PREFETCH_L1(p)  _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#  define FLAME_PREFETCH_L2(p)  _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1)
#  define FLAME_PREFETCH_NT(p)  _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_NTA)
#else
#  define RESTRICT
#  define FORCE_INLINE  inline
#  define FLAME_PREFETCH_L1(p)  ((void)(p))
#  define FLAME_PREFETCH_L2(p)  ((void)(p))
#  define FLAME_PREFETCH_NT(p)  ((void)(p))
#endif


#if defined(__AVX__) || defined(__AVX2__)
#  define FLAME_AVX 1
using simd_t = __m256;
#  define SIMD_LOAD(p)      _mm256_load_ps(p)
#  define SIMD_LOADU(p)     _mm256_loadu_ps(p)
#  define SIMD_STORE(p,r)   _mm256_store_ps(p, r)
#  define SIMD_STOREU(p,r)  _mm256_storeu_ps(p, r)
#  define SIMD_ADD(a,b)     _mm256_add_ps(a, b)
#  define SIMD_SUB(a,b)     _mm256_sub_ps(a, b)
#  define SIMD_MUL(a,b)     _mm256_mul_ps(a, b)
#  define SIMD_DIV(a,b)     _mm256_div_ps(a, b)
#  define SIMD_SET1(f)      _mm256_set1_ps(f)
#  define SIMD_ZERO()       _mm256_setzero_ps()
#  define SIMD_MIN(a,b)     _mm256_min_ps(a, b)
#  define SIMD_MAX(a,b)     _mm256_max_ps(a, b)
#  define SIMD_WIDTH        8
constexpr size_t ALIGNMENT = 32;
#else
#  define FLAME_AVX 0
using simd_t = __m128;
#  define SIMD_LOAD(p)      _mm_load_ps(p)
#  define SIMD_LOADU(p)     _mm_loadu_ps(p)
#  define SIMD_STORE(p,r)   _mm_store_ps(p, r)
#  define SIMD_STOREU(p,r)  _mm_storeu_ps(p, r)
#  define SIMD_ADD(a,b)     _mm_add_ps(a, b)
#  define SIMD_SUB(a,b)     _mm_sub_ps(a, b)
#  define SIMD_MUL(a,b)     _mm_mul_ps(a, b)
#  define SIMD_DIV(a,b)     _mm_div_ps(a, b)
#  define SIMD_SET1(f)      _mm_set1_ps(f)
#  define SIMD_ZERO()       _mm_setzero_ps()
#  define SIMD_MIN(a,b)     _mm_min_ps(a, b)
#  define SIMD_MAX(a,b)     _mm_max_ps(a, b)
#  define SIMD_WIDTH        4
constexpr size_t ALIGNMENT = 16;
#endif

namespace flame {

// ===========================================================================
// Constants
// ===========================================================================
constexpr float PI        = 3.14159265358979323846f;
constexpr float TWO_PI    = 6.28318530717958647692f;
constexpr float HALF_PI   = 1.57079632679489661923f;
constexpr float INV_PI    = 0.31830988618379067154f;
constexpr float DEG2RAD   = PI / 180.0f;
constexpr float RAD2DEG   = 180.0f / PI;
constexpr float EPSILON   = 1e-6f;
constexpr float F32_MAX   = FLT_MAX;
constexpr float SQRT2     = 1.41421356237309504880f;
constexpr float INV_SQRT2 = 0.70710678118654752440f;
constexpr float GOLDEN    = 1.61803398874989484820f;

// ===========================================================================
// Scalar utilities
// ===========================================================================
FORCE_INLINE float ToRadians(float d)                            noexcept { return d * DEG2RAD; }
FORCE_INLINE float ToDegrees(float r)                            noexcept { return r * RAD2DEG; }
FORCE_INLINE float Clampf(float v, float lo, float hi)           noexcept { return v < lo ? lo : (v > hi ? hi : v); }
FORCE_INLINE float Lerpf(float a, float b, float t)              noexcept { return a + t * (b - a); }
FORCE_INLINE float Saturate(float v)                             noexcept { return Clampf(v, 0.f, 1.f); }
FORCE_INLINE float Sign(float v)                                 noexcept { return (v > 0.f) - (v < 0.f); }
FORCE_INLINE float Frac(float v)                                 noexcept { return v - std::floor(v); }
FORCE_INLINE float Mod(float x, float y)                         noexcept { return x - y * std::floor(x / y); }
FORCE_INLINE float Smoothstep(float lo, float hi, float x)       noexcept {
    float t = Saturate((x - lo) / (hi - lo));
    return t * t * (3.f - 2.f * t);
}
FORCE_INLINE float Smootherstep(float lo, float hi, float x)     noexcept {
    float t = Saturate((x - lo) / (hi - lo));
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}
FORCE_INLINE float Remap(float v, float inLo, float inHi, float outLo, float outHi) noexcept {
    return outLo + (v - inLo) / (inHi - inLo) * (outHi - outLo);
}
FORCE_INLINE bool  NearlyEqual(float a, float b, float eps = EPSILON) noexcept {
    return std::abs(a - b) <= eps;
}
FORCE_INLINE float SafeSqrt(float v)  noexcept { return std::sqrt(v < 0.f ? 0.f : v); }
FORCE_INLINE float SafeAcos(float v)  noexcept { return std::acos(Clampf(v, -1.f, 1.f)); }
FORCE_INLINE float SafeAsin(float v)  noexcept { return std::asin(Clampf(v, -1.f, 1.f)); }

// ---------------------------------------------------------------------------
// Lane extraction (debug / serialization only — not in hot paths)
// ---------------------------------------------------------------------------
FORCE_INLINE float LaneF(const __m128 r, int i) noexcept {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, r);
    return tmp[i];
}
struct alignas(16) Vec2 {
    float _x, _y, _z, _w;   // _z/_w = padding, always 0
    FORCE_INLINE constexpr Vec2(float x = 0.f, float y = 0.f) noexcept
        : _x(x), _y(y), _z(0.f), _w(0.f) {}
};

struct alignas(16) Vec3 {
    float _x, _y, _z, _w;   // _w = padding, always 0
    FORCE_INLINE constexpr Vec3(float x = 0.f, float y = 0.f, float z = 0.f) noexcept
        : _x(x), _y(y), _z(z), _w(0.f) {}
};

struct alignas(16) Vec4 {
    float _x, _y, _z, _w;
    FORCE_INLINE constexpr Vec4(float x = 0.f, float y = 0.f,
                                float z = 0.f, float w = 1.f) noexcept
        : _x(x), _y(y), _z(z), _w(w) {}
};

struct alignas(16) Quat {
    float _x, _y, _z, _w;
    FORCE_INLINE constexpr Quat(float x = 0.f, float y = 0.f,
                                float z = 0.f, float w = 1.f) noexcept
        : _x(x), _y(y), _z(z), _w(w) {}
};


template <typename T>
concept vec_type = requires(T v) {
    { v._x } -> std::same_as<float&>;
    { v._y } -> std::same_as<float&>;
};


struct alignas(32) Vec4SoA {
    float x[8];
    float y[8];
    float z[8];
    float w[8];

    Vec4SoA() noexcept {
        for (int i = 0; i < 8; ++i) x[i] = y[i] = z[i] = 0.f, w[i] = 1.f;
    }
    FORCE_INLINE void Set(int lane, const Vec4& v) noexcept {
        assert(lane >= 0 && lane < 8);
        x[lane] = v._x; y[lane] = v._y;
        z[lane] = v._z; w[lane] = v._w;
    }
    FORCE_INLINE Vec4 Get(int lane) const noexcept {
        assert(lane >= 0 && lane < 8);
        return { x[lane], y[lane], z[lane], w[lane] };
    }
};

FORCE_INLINE Vec4SoA AddSoA(const Vec4SoA& a, const Vec4SoA& b) noexcept {
    Vec4SoA out;
#if FLAME_AVX
    _mm256_store_ps(out.x, _mm256_add_ps(_mm256_load_ps(a.x), _mm256_load_ps(b.x)));
    _mm256_store_ps(out.y, _mm256_add_ps(_mm256_load_ps(a.y), _mm256_load_ps(b.y)));
    _mm256_store_ps(out.z, _mm256_add_ps(_mm256_load_ps(a.z), _mm256_load_ps(b.z)));
    _mm256_store_ps(out.w, _mm256_add_ps(_mm256_load_ps(a.w), _mm256_load_ps(b.w)));
#else
    _mm_store_ps(out.x,     _mm_add_ps(_mm_load_ps(a.x),     _mm_load_ps(b.x)));
    _mm_store_ps(out.x + 4, _mm_add_ps(_mm_load_ps(a.x + 4), _mm_load_ps(b.x + 4)));
    _mm_store_ps(out.y,     _mm_add_ps(_mm_load_ps(a.y),     _mm_load_ps(b.y)));
    _mm_store_ps(out.y + 4, _mm_add_ps(_mm_load_ps(a.y + 4), _mm_load_ps(b.y + 4)));
    _mm_store_ps(out.z,     _mm_add_ps(_mm_load_ps(a.z),     _mm_load_ps(b.z)));
    _mm_store_ps(out.z + 4, _mm_add_ps(_mm_load_ps(a.z + 4), _mm_load_ps(b.z + 4)));
    _mm_store_ps(out.w,     _mm_add_ps(_mm_load_ps(a.w),     _mm_load_ps(b.w)));
    _mm_store_ps(out.w + 4, _mm_add_ps(_mm_load_ps(a.w + 4), _mm_load_ps(b.w + 4)));
#endif
    return out;
}

FORCE_INLINE Vec4SoA MulSoA(const Vec4SoA& a, float s) noexcept {
    Vec4SoA out;
#if FLAME_AVX
    __m256 vs = _mm256_set1_ps(s);
    _mm256_store_ps(out.x, _mm256_mul_ps(_mm256_load_ps(a.x), vs));
    _mm256_store_ps(out.y, _mm256_mul_ps(_mm256_load_ps(a.y), vs));
    _mm256_store_ps(out.z, _mm256_mul_ps(_mm256_load_ps(a.z), vs));
    _mm256_store_ps(out.w, _mm256_mul_ps(_mm256_load_ps(a.w), vs));
#else
    __m128 vs = _mm_set1_ps(s);
    _mm_store_ps(out.x,     _mm_mul_ps(_mm_load_ps(a.x),     vs));
    _mm_store_ps(out.x + 4, _mm_mul_ps(_mm_load_ps(a.x + 4), vs));
    _mm_store_ps(out.y,     _mm_mul_ps(_mm_load_ps(a.y),     vs));
    _mm_store_ps(out.y + 4, _mm_mul_ps(_mm_load_ps(a.y + 4), vs));
    _mm_store_ps(out.z,     _mm_mul_ps(_mm_load_ps(a.z),     vs));
    _mm_store_ps(out.z + 4, _mm_mul_ps(_mm_load_ps(a.z + 4), vs));
    _mm_store_ps(out.w,     _mm_mul_ps(_mm_load_ps(a.w),     vs));
    _mm_store_ps(out.w + 4, _mm_mul_ps(_mm_load_ps(a.w + 4), vs));
#endif
    return out;
}

FORCE_INLINE void DotSoA(const Vec4SoA& a, const Vec4SoA& b,
                         float* RESTRICT out8) noexcept {
#if FLAME_AVX
    __m256 ax = _mm256_load_ps(a.x), bx = _mm256_load_ps(b.x);
    __m256 ay = _mm256_load_ps(a.y), by = _mm256_load_ps(b.y);
    __m256 az = _mm256_load_ps(a.z), bz = _mm256_load_ps(b.z);
    __m256 d = _mm256_fmadd_ps(ax, bx,
               _mm256_fmadd_ps(ay, by, _mm256_mul_ps(az, bz)));
    _mm256_store_ps(out8, d);
#else
    __m128 ax0 = _mm_load_ps(a.x),     bx0 = _mm_load_ps(b.x);
    __m128 ax1 = _mm_load_ps(a.x + 4), bx1 = _mm_load_ps(b.x + 4);
    __m128 ay0 = _mm_load_ps(a.y),     by0 = _mm_load_ps(b.y);
    __m128 ay1 = _mm_load_ps(a.y + 4), by1 = _mm_load_ps(b.y + 4);
    __m128 az0 = _mm_load_ps(a.z),     bz0 = _mm_load_ps(b.z);
    __m128 az1 = _mm_load_ps(a.z + 4), bz1 = _mm_load_ps(b.z + 4);
    _mm_store_ps(out8,
        _mm_add_ps(_mm_mul_ps(ax0, bx0),
        _mm_add_ps(_mm_mul_ps(ay0, by0), _mm_mul_ps(az0, bz0))));
    _mm_store_ps(out8 + 4,
        _mm_add_ps(_mm_mul_ps(ax1, bx1),
        _mm_add_ps(_mm_mul_ps(ay1, by1), _mm_mul_ps(az1, bz1))));
#endif
}



template <vec_type T>
FORCE_INLINE T ADD(const T& a, const T& b) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_add_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
    return out;
}

template <vec_type T>
FORCE_INLINE T SUB(const T& a, const T& b) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_sub_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
    return out;
}

template <vec_type T>
FORCE_INLINE T MUL(const T& a, float s) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_mul_ps(_mm_load_ps(&a._x), _mm_set1_ps(s)));
    return out;
}

template <vec_type T>
FORCE_INLINE T DIV(const T& a, float s) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_div_ps(_mm_load_ps(&a._x), _mm_set1_ps(s)));
    return out;
}

template <vec_type T>
FORCE_INLINE T LERP(const T& a, const T& b, float t) noexcept {
    T out;
    __m128 vt = _mm_set1_ps(t);
    __m128 va = _mm_load_ps(&a._x);
    __m128 vb = _mm_load_ps(&b._x);
    _mm_store_ps(&out._x, _mm_add_ps(va, _mm_mul_ps(vt, _mm_sub_ps(vb, va))));
    return out;
}

template <vec_type T>
FORCE_INLINE T CLAMP(const T& v, float lo, float hi) noexcept {
    T out;
    _mm_store_ps(&out._x,
        _mm_min_ps(_mm_max_ps(_mm_load_ps(&v._x), _mm_set1_ps(lo)),
                   _mm_set1_ps(hi)));
    return out;
}

template <vec_type T>
FORCE_INLINE T VMIN(const T& a, const T& b) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_min_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
    return out;
}

template <vec_type T>
FORCE_INLINE T VMAX(const T& a, const T& b) noexcept {
    T out;
    _mm_store_ps(&out._x, _mm_max_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
    return out;
}

template <vec_type T>
FORCE_INLINE T NEG(const T& a) noexcept {
    T out;
    _mm_store_ps(&out._x,
        _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(&a._x)));
    return out;
}

// DOT — uses xyz lanes only (mask 0x71: read xyz, write lane0)
template <vec_type T>
FORCE_INLINE float DOT(const T& a, const T& b) noexcept {
    return _mm_cvtss_f32(
        _mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x), 0x71));
}

// DOT4 — full xyzw
FORCE_INLINE float DOT4(const Vec4& a, const Vec4& b) noexcept {
    return _mm_cvtss_f32(
        _mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x), 0xF1));
}

// LENGTH — using xyz lanes
template <vec_type T>
FORCE_INLINE float LENGTH(const T& a) noexcept {
    __m128 reg = _mm_load_ps(&a._x);
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(reg, reg, 0x71)));
}

FORCE_INLINE float LENGTH4(const Vec4& a) noexcept {
    __m128 reg = _mm_load_ps(&a._x);
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(reg, reg, 0xF1)));
}

FORCE_INLINE float LENGTH_SQ(const Vec3& a) noexcept {
    return _mm_cvtss_f32(_mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&a._x), 0x71));
}

// NORMALIZE — Newton-Raphson rsqrt (one NR step, ~23-bit accuracy)
// Dot mask 0x7F: read xyz, broadcast result to all lanes → _w gets normalized too
// but since _w=0, 0*inv=0 preserves the padding invariant.
template <vec_type T>
FORCE_INLINE T NORMALIZE(const T& a) noexcept {
    __m128 reg = _mm_load_ps(&a._x);
    __m128 lsq = _mm_dp_ps(reg, reg, 0x7F);
    __m128 est = _mm_rsqrt_ps(lsq);
    __m128 half  = _mm_set1_ps(0.5f);
    __m128 three = _mm_set1_ps(3.0f);
    // NR: est = 0.5 * est * (3 - lsq * est * est)
    __m128 inv = _mm_mul_ps(half,
                    _mm_mul_ps(est,
                        _mm_sub_ps(three, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
    T out;
    _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
    return out;
}

FORCE_INLINE Vec4 NORMALIZE4(const Vec4& a) noexcept {
    __m128 reg = _mm_load_ps(&a._x);
    __m128 lsq = _mm_dp_ps(reg, reg, 0xFF);
    __m128 est = _mm_rsqrt_ps(lsq);
    __m128 half  = _mm_set1_ps(0.5f);
    __m128 three = _mm_set1_ps(3.0f);
    __m128 inv = _mm_mul_ps(half,
                    _mm_mul_ps(est,
                        _mm_sub_ps(three, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
    Vec4 out;
    _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
    return out;
}

// Safe normalize — returns zero vector if length < epsilon
template <vec_type T>
FORCE_INLINE T NORMALIZE_SAFE(const T& a, const T& fallback = T{}) noexcept {
    float lsq = LENGTH_SQ(static_cast<const Vec3&>(a));
    if (lsq < EPSILON * EPSILON) return fallback;
    return NORMALIZE(a);
}

FORCE_INLINE Vec3 CROSS(const Vec3& a, const Vec3& b) noexcept {
    __m128 rA = _mm_load_ps(&a._x);
    __m128 rB = _mm_load_ps(&b._x);
    // a.yzx × b.zxy − a.zxy × b.yzx
    __m128 t0 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 0, 2, 1)); // a.y a.z a.x _
    __m128 t1 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 1, 0, 2)); // b.z b.x b.y _
    __m128 t2 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 1, 0, 2)); // a.z a.x a.y _
    __m128 t3 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 0, 2, 1)); // b.y b.z b.x _
    Vec3 out;
    _mm_store_ps(&out._x, _mm_sub_ps(_mm_mul_ps(t0, t1), _mm_mul_ps(t2, t3)));
    return out;
}

FORCE_INLINE Vec3 REFLECT(const Vec3& v, const Vec3& n) noexcept {
    return SUB(v, MUL(n, 2.f * DOT(v, n)));
}

FORCE_INLINE Vec3 REFRACT(const Vec3& v, const Vec3& n, float eta) noexcept {
    float cosI = -DOT(v, n);
    float sinT2 = eta * eta * (1.f - cosI * cosI);
    if (sinT2 > 1.f) return Vec3(0, 0, 0);
    float cosT = std::sqrt(1.f - sinT2);
    return ADD(MUL(v, eta), MUL(n, eta * cosI - cosT));
}

template <vec_type T>
FORCE_INLINE float DISTANCE(const T& a, const T& b) noexcept {
    return LENGTH(SUB(a, b));
}

FORCE_INLINE float DISTANCE_SQ(const Vec3& a, const Vec3& b) noexcept {
    Vec3 d = SUB(a, b);
    return LENGTH_SQ(d);
}

FORCE_INLINE Vec3 PROJECT(const Vec3& v, const Vec3& onto) noexcept {
    return MUL(onto, DOT(v, onto) / DOT(onto, onto));
}

FORCE_INLINE Vec3 REJECT(const Vec3& v, const Vec3& onto) noexcept {
    return SUB(v, PROJECT(v, onto));
}

// Angle between two unit vectors (in radians)
FORCE_INLINE float ANGLE(const Vec3& a, const Vec3& b) noexcept {
    return SafeAcos(DOT(a, b));
}

// ===========================================================================
// Quaternion operations
// ===========================================================================
FORCE_INLINE Quat QuatIdentity()              noexcept { return { 0.f, 0.f, 0.f, 1.f }; }
FORCE_INLINE bool QuatIsUnit(const Quat& q)   noexcept {
    float lsq = q._x*q._x + q._y*q._y + q._z*q._z + q._w*q._w;
    return NearlyEqual(lsq, 1.f, EPSILON);
}

FORCE_INLINE Quat QuatNormalize(const Quat& q) noexcept {
    __m128 reg = _mm_load_ps(&q._x);
    __m128 lsq = _mm_dp_ps(reg, reg, 0xFF);
    __m128 est = _mm_rsqrt_ps(lsq);
    __m128 half  = _mm_set1_ps(0.5f);
    __m128 thr   = _mm_set1_ps(3.0f);
    __m128 inv = _mm_mul_ps(half,
                    _mm_mul_ps(est,
                        _mm_sub_ps(thr, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
    Quat out;
    _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
    return out;
}

FORCE_INLINE Quat QuatConjugate(const Quat& q) noexcept {
    // Flip xyz sign bits, leave w untouched
    // _mm_set_ps(w, z, y, x) — reverse order: e3=w=0, e2=z=-0, e1=y=-0, e0=x=-0
    static const __m128 SIGN_MASK = _mm_set_ps(0.f, -0.f, -0.f, -0.f);
    Quat out;
    _mm_store_ps(&out._x, _mm_xor_ps(_mm_load_ps(&q._x), SIGN_MASK));
    return out;
}

FORCE_INLINE Quat QuatInverse(const Quat& q) noexcept {
    // For unit quaternions, conjugate == inverse; use normalize for safety
    return QuatNormalize(QuatConjugate(q));
}

FORCE_INLINE Quat QuatMul(const Quat& a, const Quat& b) noexcept {
    float ax = a._x, ay = a._y, az = a._z, aw = a._w;
    float bx = b._x, by = b._y, bz = b._z, bw = b._w;
    return {
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz
    };
}

FORCE_INLINE Quat QuatFromAxisAngle(const Vec3& axis, float angle) noexcept {
    float half = angle * 0.5f;
    float s = std::sin(half), c = std::cos(half);
    return { axis._x * s, axis._y * s, axis._z * s, c };
}

// Euler angles in radians (pitch=X, yaw=Y, roll=Z), applied in ZYX order
FORCE_INLINE Quat QuatFromEuler(float pitchRad, float yawRad, float rollRad) noexcept {
    float hp = pitchRad * 0.5f, hy = yawRad * 0.5f, hr = rollRad * 0.5f;
    float cp = std::cos(hp), sp = std::sin(hp);
    float cy = std::cos(hy), sy = std::sin(hy);
    float cr = std::cos(hr), sr = std::sin(hr);
    return {
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    };
}

// Returns (pitch, yaw, roll) in radians
FORCE_INLINE Vec3 QuatToEuler(const Quat& q) noexcept {
    float x = q._x, y = q._y, z = q._z, w = q._w;
    // Pitch (x-axis rotation)
    float sinP = 2.f * (w * x + y * z);
    float cosP = 1.f - 2.f * (x * x + y * y);
    float pitch = std::atan2(sinP, cosP);
    // Yaw (y-axis rotation)
    float sinY = 2.f * (w * y - z * x);
    float yaw = (std::abs(sinY) >= 1.f) ? std::copysign(HALF_PI, sinY) : std::asin(sinY);
    // Roll (z-axis rotation)
    float sinR = 2.f * (w * z + x * y);
    float cosR = 1.f - 2.f * (y * y + z * z);
    float roll = std::atan2(sinR, cosR);
    return { pitch, yaw, roll };
}

FORCE_INLINE Vec3 QuatRotate(const Quat& q, const Vec3& v) noexcept {
    Vec3 qv = { q._x, q._y, q._z };
    Vec3 t = MUL(CROSS(qv, v), 2.f);
    return ADD(ADD(v, MUL(t, q._w)), CROSS(qv, t));
}

// Dot product of two quaternions (for slerp / angle between)
FORCE_INLINE float QuatDot(const Quat& a, const Quat& b) noexcept {
    return a._x*b._x + a._y*b._y + a._z*b._z + a._w*b._w;
}

FORCE_INLINE float QuatAngle(const Quat& a, const Quat& b) noexcept {
    float d = QuatDot(a, b);
    return 2.f * SafeAcos(std::abs(d));
}

FORCE_INLINE Quat QuatSlerp(const Quat& a, const Quat& b, float t) noexcept {
    float dot = QuatDot(a, b);
    Quat bAdj = b;
    if (dot < 0.f) { dot = -dot; bAdj = { -b._x, -b._y, -b._z, -b._w }; }

    if (dot > 0.9995f) {
        // Nearly identical → normalized lerp
        __m128 va  = _mm_load_ps(&a._x);
        __m128 vb  = _mm_load_ps(&bAdj._x);
        __m128 vt  = _mm_set1_ps(t);
        __m128 res = _mm_add_ps(va, _mm_mul_ps(vt, _mm_sub_ps(vb, va)));
        __m128 lsq = _mm_dp_ps(res, res, 0xFF);
        __m128 est = _mm_rsqrt_ps(lsq);
        __m128 hv  = _mm_set1_ps(0.5f), tv = _mm_set1_ps(3.0f);
        __m128 inv = _mm_mul_ps(hv, _mm_mul_ps(est,
                         _mm_sub_ps(tv, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
        Quat out;
        _mm_store_ps(&out._x, _mm_mul_ps(res, inv));
        return out;
    }
    float theta0    = std::acos(dot);
    float sinTheta0 = std::sin(theta0);
    float theta     = theta0 * t;
    float sinTheta  = std::sin(theta);
    float cosTheta  = std::cos(theta);
    float s0 = cosTheta - dot * sinTheta / sinTheta0;
    float s1 = sinTheta / sinTheta0;

    __m128 va = _mm_load_ps(&a._x);
    __m128 vb = _mm_load_ps(&bAdj._x);
    Quat out;
    _mm_store_ps(&out._x,
        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(s0), va),
                   _mm_mul_ps(_mm_set1_ps(s1), vb)));
    return out;
}

// Forward-declare so QuatToMat4 free-function can reference Mat4
FORCE_INLINE void QuatToMat4(const Quat& q, struct Mat4& out) noexcept;


struct alignas(16) Mat4 {
    __m128 rows[4];

    static FORCE_INLINE Mat4 Identity() noexcept {
        Mat4 m;
        m.rows[0] = _mm_setr_ps(1, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, 1, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, 1, 0);
        m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
        return m;
    }

    static FORCE_INLINE Mat4 Zero() noexcept {
        Mat4 m;
        m.rows[0] = m.rows[1] = m.rows[2] = m.rows[3] = _mm_setzero_ps();
        return m;
    }

    FORCE_INLINE float Get(int row, int col) const noexcept {
        return LaneF(rows[row], col);
    }
    FORCE_INLINE void Set(int row, int col, float v) noexcept {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, rows[row]);
        tmp[col] = v;
        rows[row] = _mm_load_ps(tmp);
    }

    // Raw float pointer (column-major reinterpretation needs Transpose first)
    FORCE_INLINE const float* Data() const noexcept {
        return reinterpret_cast<const float*>(rows);
    }
};


FORCE_INLINE Mat4 MUL(const Mat4& a, const Mat4& b) noexcept {
    Mat4 out;
    for (int i = 0; i < 4; ++i) {
        __m128 row = a.rows[i];
        __m128 res = _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(0,0,0,0)), b.rows[0]);
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(1,1,1,1)), b.rows[1]));
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(2,2,2,2)), b.rows[2]));
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(3,3,3,3)), b.rows[3]));
        out.rows[i] = res;
    }
    return out;
}

FORCE_INLINE Vec4 TransformVec4(const Mat4& m, const Vec4& v) noexcept {
    __m128 vv  = _mm_load_ps(&v._x);
    __m128 res = _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(0,0,0,0)), m.rows[0]);
    res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(1,1,1,1)), m.rows[1]));
    res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(2,2,2,2)), m.rows[2]));
    res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(3,3,3,3)), m.rows[3]));
    Vec4 out;
    _mm_store_ps(&out._x, res);
    return out;
}

FORCE_INLINE Vec3 TransformPoint(const Mat4& m, const Vec3& p) noexcept {
    Vec4 v4 = { p._x, p._y, p._z, 1.f };
    Vec4 r  = TransformVec4(m, v4);
    float invW = 1.f / r._w;
    return { r._x * invW, r._y * invW, r._z * invW };
}

FORCE_INLINE Vec3 TransformDir(const Mat4& m, const Vec3& d) noexcept {
    Vec4 v4 = { d._x, d._y, d._z, 0.f };
    Vec4 r  = TransformVec4(m, v4);
    return { r._x, r._y, r._z };
}

FORCE_INLINE Mat4 Transpose(const Mat4& m) noexcept {
    Mat4 out = m;
    _MM_TRANSPOSE4_PS(out.rows[0], out.rows[1], out.rows[2], out.rows[3]);
    return out;
}

FORCE_INLINE Mat4 Inverse(const Mat4& src) noexcept {
    // SSE matrix inverse — Intel's classic 4-wide cofactor algorithm
    Mat4 m = Transpose(src);
    __m128 row0 = m.rows[0], row1 = m.rows[1];
    __m128 row2 = m.rows[2], row3 = m.rows[3];
    __m128 tmp1, minor0, minor1, minor2, minor3, det;

    tmp1   = _mm_mul_ps(row2, row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_mul_ps(row1, tmp1);
    minor1 = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

    tmp1   = _mm_mul_ps(row1, row2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3 = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

    tmp1   = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2   = _mm_shuffle_ps(row2, row2, 0x4E);
    minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2 = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

    tmp1   = _mm_mul_ps(row0, row1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

    tmp1   = _mm_mul_ps(row0, row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

    tmp1   = _mm_mul_ps(row0, row2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

    det    = _mm_mul_ps(row0, minor0);
    det    = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det    = _mm_add_ps(_mm_shuffle_ps(det, det, 0xB1), det);
    tmp1   = _mm_rcp_ps(det);
    det    = _mm_sub_ps(_mm_add_ps(tmp1, tmp1),
                        _mm_mul_ps(det, _mm_mul_ps(tmp1, tmp1)));

    Mat4 out;
    out.rows[0] = _mm_mul_ps(det, minor0);
    out.rows[1] = _mm_mul_ps(det, minor1);
    out.rows[2] = _mm_mul_ps(det, minor2);
    out.rows[3] = _mm_mul_ps(det, minor3);
    return out;
}

FORCE_INLINE float Determinant(const Mat4& m) noexcept {
    float a   = m.Get(0,0), b   = m.Get(0,1), c   = m.Get(0,2), d   = m.Get(0,3);
    float m10 = m.Get(1,0), m11 = m.Get(1,1), m12 = m.Get(1,2), m13 = m.Get(1,3);
    float m20 = m.Get(2,0), m21 = m.Get(2,1), m22 = m.Get(2,2), m23 = m.Get(2,3);
    float m30 = m.Get(3,0), m31 = m.Get(3,1), m32 = m.Get(3,2), m33 = m.Get(3,3);
    float C0 = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
    float C1 = m10*(m22*m33 - m23*m32) - m12*(m20*m33 - m23*m30) + m13*(m20*m32 - m22*m30);
    float C2 = m10*(m21*m33 - m23*m31) - m11*(m20*m33 - m23*m30) + m13*(m20*m31 - m21*m30);
    float C3 = m10*(m21*m32 - m22*m31) - m11*(m20*m32 - m22*m30) + m12*(m20*m31 - m21*m30);
    return a*C0 - b*C1 + c*C2 - d*C3;
}


FORCE_INLINE Mat4 Translate(const Vec3& v) noexcept {
    Mat4 m = Mat4::Identity();
    m.rows[0] = _mm_setr_ps(1, 0, 0, v._x);
    m.rows[1] = _mm_setr_ps(0, 1, 0, v._y);
    m.rows[2] = _mm_setr_ps(0, 0, 1, v._z);
    return m;
}

FORCE_INLINE Mat4 Scale(const Vec3& v) noexcept {
    Mat4 m = Mat4::Identity();
    m.rows[0] = _mm_setr_ps(v._x, 0, 0, 0);
    m.rows[1] = _mm_setr_ps(0, v._y, 0, 0);
    m.rows[2] = _mm_setr_ps(0, 0, v._z, 0);
    return m;
}

FORCE_INLINE Mat4 Scale(float s) noexcept { return Scale({ s, s, s }); }

FORCE_INLINE Mat4 RotateX(float angle) noexcept {
    float s = std::sin(angle), c = std::cos(angle);
    Mat4 m = Mat4::Identity();
    m.rows[1] = _mm_setr_ps(0,  c, -s, 0);
    m.rows[2] = _mm_setr_ps(0,  s,  c, 0);
    return m;
}

FORCE_INLINE Mat4 RotateY(float angle) noexcept {
    float s = std::sin(angle), c = std::cos(angle);
    Mat4 m = Mat4::Identity();
    m.rows[0] = _mm_setr_ps( c, 0, s, 0);
    m.rows[2] = _mm_setr_ps(-s, 0, c, 0);
    return m;
}

FORCE_INLINE Mat4 RotateZ(float angle) noexcept {
    float s = std::sin(angle), c = std::cos(angle);
    Mat4 m = Mat4::Identity();
    m.rows[0] = _mm_setr_ps( c, -s, 0, 0);
    m.rows[1] = _mm_setr_ps( s,  c, 0, 0);
    return m;
}

FORCE_INLINE Mat4 RotateAxisAngle(const Vec3& axis, float angle) noexcept {
    Vec3 u = NORMALIZE(axis);
    float s = std::sin(angle), c = std::cos(angle), ic = 1.f - c;
    float x = u._x, y = u._y, z = u._z;
    Mat4 m = Mat4::Identity();
    m.rows[0] = _mm_setr_ps(c + x*x*ic,     x*y*ic - z*s,  x*z*ic + y*s, 0);
    m.rows[1] = _mm_setr_ps(y*x*ic + z*s,   c + y*y*ic,    y*z*ic - x*s, 0);
    m.rows[2] = _mm_setr_ps(z*x*ic - y*s,   z*y*ic + x*s,  c + z*z*ic,   0);
    return m;
}

FORCE_INLINE void QuatToMat4(const Quat& q, Mat4& out) noexcept {
    float x = q._x, y = q._y, z = q._z, w = q._w;
    float x2 = x+x, y2 = y+y, z2 = z+z;
    float xx = x*x2, xy = x*y2, xz = x*z2;
    float yy = y*y2, yz = y*z2, zz = z*z2;
    float wx = w*x2, wy = w*y2, wz = w*z2;
    out = Mat4::Identity();
    out.rows[0] = _mm_setr_ps(1-(yy+zz), xy-wz,    xz+wy,    0);
    out.rows[1] = _mm_setr_ps(xy+wz,     1-(xx+zz), yz-wx,   0);
    out.rows[2] = _mm_setr_ps(xz-wy,     yz+wx,     1-(xx+yy), 0);
}

FORCE_INLINE Mat4 QuatToMat4(const Quat& q) noexcept {
    Mat4 out; QuatToMat4(q, out); return out;
}

FORCE_INLINE Mat4 TRS(const Vec3& t, const Quat& r, const Vec3& s) noexcept {
    return MUL(MUL(Translate(t), QuatToMat4(r)), Scale(s));
}

// Depth range [0,1] (DX convention), right-handed, looking down +Z
FORCE_INLINE Mat4 Perspective(float fovDeg, float aspect,
                               float nearP, float farP) noexcept {
    float f     = 1.f / std::tan(ToRadians(fovDeg) * 0.5f);
    float range = farP - nearP;
    Mat4 m;
    m.rows[0] = _mm_setr_ps(f / aspect, 0, 0, 0);
    m.rows[1] = _mm_setr_ps(0, f, 0, 0);
    m.rows[2] = _mm_setr_ps(0, 0, farP / range, -(farP * nearP) / range);
    m.rows[3] = _mm_setr_ps(0, 0, 1, 0);
    return m;
}

// Reversed-Z variant [1,0] — better floating-point precision for distant geometry
FORCE_INLINE Mat4 PerspectiveRevZ(float fovDeg, float aspect,
                                   float nearP, float farP) noexcept {
    float f     = 1.f / std::tan(ToRadians(fovDeg) * 0.5f);
    float range = nearP - farP;
    Mat4 m;
    m.rows[0] = _mm_setr_ps(f / aspect, 0, 0, 0);
    m.rows[1] = _mm_setr_ps(0, f, 0, 0);
    m.rows[2] = _mm_setr_ps(0, 0, nearP / range, -(nearP * farP) / range);
    m.rows[3] = _mm_setr_ps(0, 0, 1, 0);
    return m;
}

FORCE_INLINE Mat4 Orthographic(float left, float right,
                                float bottom, float top,
                                float nearP, float farP) noexcept {
    float rml = right - left, rpl = right + left;
    float tmb = top - bottom, tpb = top + bottom;
    float fmn = farP - nearP, fpn = farP + nearP;
    Mat4 m;
    m.rows[0] = _mm_setr_ps(2/rml, 0,     0,     -rpl/rml);
    m.rows[1] = _mm_setr_ps(0,     2/tmb, 0,     -tpb/tmb);
    m.rows[2] = _mm_setr_ps(0,     0,     2/fmn, -fpn/fmn);
    m.rows[3] = _mm_setr_ps(0,     0,     0,      1);
    return m;
}

FORCE_INLINE Mat4 LookAt(const Vec3& eye, const Vec3& target,
                          const Vec3& up) noexcept {
    Vec3 z = NORMALIZE(SUB(target, eye));         // forward
    Vec3 x = NORMALIZE(CROSS(z, up));             // right
    Vec3 y = CROSS(x, z);                         // true up
    Mat4 m;
    m.rows[0] = _mm_setr_ps( x._x,  x._y,  x._z, -DOT(x, eye));
    m.rows[1] = _mm_setr_ps( y._x,  y._y,  y._z, -DOT(y, eye));
    m.rows[2] = _mm_setr_ps(-z._x, -z._y, -z._z,  DOT(z, eye));
    m.rows[3] = _mm_setr_ps( 0,     0,     0,      1);
    return m;
}


struct alignas(16) Mat3 {
    __m128 rows[3]; // rows[i] = (m[i][0], m[i][1], m[i][2], 0)

    static FORCE_INLINE Mat3 Identity() noexcept {
        Mat3 m;
        m.rows[0] = _mm_setr_ps(1, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, 1, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, 1, 0);
        return m;
    }
    static FORCE_INLINE Mat3 Zero() noexcept {
        Mat3 m;
        m.rows[0] = m.rows[1] = m.rows[2] = _mm_setzero_ps();
        return m;
    }
    FORCE_INLINE float Get(int row, int col) const noexcept { return LaneF(rows[row], col); }
    FORCE_INLINE void  Set(int row, int col, float v) noexcept {
        alignas(16) float tmp[4]; _mm_store_ps(tmp, rows[row]);
        tmp[col] = v; rows[row] = _mm_load_ps(tmp);
    }
};

FORCE_INLINE Vec3 MUL(const Mat3& m, const Vec3& v) noexcept {
    __m128 vv  = _mm_load_ps(&v._x);
    __m128 res = _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(0,0,0,0)), m.rows[0]);
    res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(1,1,1,1)), m.rows[1]));
    res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv,vv,_MM_SHUFFLE(2,2,2,2)), m.rows[2]));
    Vec3 out; _mm_store_ps(&out._x, res); out._w = 0.f;
    return out;
}

FORCE_INLINE Mat3 MUL(const Mat3& a, const Mat3& b) noexcept {
    Mat3 out;
    for (int i = 0; i < 3; ++i) {
        __m128 row = a.rows[i];
        __m128 res = _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(0,0,0,0)), b.rows[0]);
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(1,1,1,1)), b.rows[1]));
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row,row,_MM_SHUFFLE(2,2,2,2)), b.rows[2]));
        // Zero out padding lane 3
        res = _mm_and_ps(res, _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,0)));
        out.rows[i] = res;
    }
    return out;
}

FORCE_INLINE Mat3 Transpose(const Mat3& m) noexcept {
    // Pad to 4×4, transpose, pull back
    __m128 r0 = m.rows[0], r1 = m.rows[1], r2 = m.rows[2], r3 = _mm_setzero_ps();
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    Mat3 out;
    out.rows[0] = _mm_and_ps(r0, _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,0)));
    out.rows[1] = _mm_and_ps(r1, _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,0)));
    out.rows[2] = _mm_and_ps(r2, _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,0)));
    return out;
}

// Extract upper-left 3×3 from a Mat4
FORCE_INLINE Mat3 Mat3FromMat4(const Mat4& m) noexcept {
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,0));
    Mat3 out;
    out.rows[0] = _mm_and_ps(m.rows[0], mask);
    out.rows[1] = _mm_and_ps(m.rows[1], mask);
    out.rows[2] = _mm_and_ps(m.rows[2], mask);
    return out;
}

// Normal matrix = transpose(inverse(upper-left 3x3 of modelMatrix))
// For uniform scale, this equals the rotation submatrix directly.
FORCE_INLINE Mat3 NormalMatrix(const Mat4& model) noexcept {
    // Compute cofactor matrix of the upper-left 3x3 (= det * inverse^T)
    // For well-conditioned transforms, extract + transpose the inverse is robust
    float m00 = model.Get(0,0), m01 = model.Get(0,1), m02 = model.Get(0,2);
    float m10 = model.Get(1,0), m11 = model.Get(1,1), m12 = model.Get(1,2);
    float m20 = model.Get(2,0), m21 = model.Get(2,1), m22 = model.Get(2,2);
    // Cofactors
    float c00 = m11*m22 - m12*m21, c01 = m12*m20 - m10*m22, c02 = m10*m21 - m11*m20;
    float c10 = m02*m21 - m01*m22, c11 = m00*m22 - m02*m20, c12 = m01*m20 - m00*m21;
    float c20 = m01*m12 - m02*m11, c21 = m02*m10 - m00*m12, c22 = m00*m11 - m01*m10;
    float det = m00*c00 + m01*c01 + m02*c02;
    float invDet = (std::abs(det) > EPSILON) ? (1.f / det) : 0.f;
    Mat3 out;
    out.rows[0] = _mm_setr_ps(c00*invDet, c01*invDet, c02*invDet, 0);
    out.rows[1] = _mm_setr_ps(c10*invDet, c11*invDet, c12*invDet, 0);
    out.rows[2] = _mm_setr_ps(c20*invDet, c21*invDet, c22*invDet, 0);
    return out;
}


struct alignas(16) Transform {
    Vec3 position   = {};
    Quat rotation   = QuatIdentity();
    Vec3 scale      = { 1.f, 1.f, 1.f };

    // Construct a local-to-world matrix
    FORCE_INLINE Mat4 ToMat4() const noexcept {
        return TRS(position, rotation, scale);
    }

    // Construct a world-to-local (view) matrix
    FORCE_INLINE Mat4 ToInverseMat4() const noexcept {
        return Inverse(ToMat4());
    }

    // Transform a world point into local space
    FORCE_INLINE Vec3 InverseTransformPoint(const Vec3& wp) const noexcept {
        Vec3 d  = SUB(wp, position);
        Vec3 ls = QuatRotate(QuatConjugate(rotation), d);
        return Vec3(ls._x / scale._x, ls._y / scale._y, ls._z / scale._z);
    }

    // Combine parent * child (apply parent transform to child)
    FORCE_INLINE Transform Combined(const Transform& child) const noexcept {
        Transform out;
        out.position = ADD(position, QuatRotate(rotation, Vec3(
            child.position._x * scale._x,
            child.position._y * scale._y,
            child.position._z * scale._z)));
        out.rotation = QuatNormalize(QuatMul(rotation, child.rotation));
        out.scale    = Vec3(scale._x * child.scale._x,
                            scale._y * child.scale._y,
                            scale._z * child.scale._z);
        return out;
    }

    FORCE_INLINE Vec3 Forward() const noexcept { return QuatRotate(rotation, {0,0,1}); }
    FORCE_INLINE Vec3 Right()   const noexcept { return QuatRotate(rotation, {1,0,0}); }
    FORCE_INLINE Vec3 Up()      const noexcept { return QuatRotate(rotation, {0,1,0}); }
};


struct alignas(16) Plane {
    float _a, _b, _c, _d;   // matches Vec4 layout for SIMD load

    static FORCE_INLINE Plane FromNormalPoint(const Vec3& n, const Vec3& p) noexcept {
        return { n._x, n._y, n._z, -DOT(n, p) };
    }

    static FORCE_INLINE Plane FromThreePoints(const Vec3& p0,
                                               const Vec3& p1,
                                               const Vec3& p2) noexcept {
        Vec3 n = NORMALIZE(CROSS(SUB(p1, p0), SUB(p2, p0)));
        return FromNormalPoint(n, p0);
    }

    // Signed distance (positive = same side as normal)
    FORCE_INLINE float SignedDistance(const Vec3& p) const noexcept {
        return _a*p._x + _b*p._y + _c*p._z + _d;
    }

    // Normalize the plane (divide by normal length)
    FORCE_INLINE Plane Normalized() const noexcept {
        float invLen = 1.f / std::sqrt(_a*_a + _b*_b + _c*_c);
        return { _a*invLen, _b*invLen, _c*invLen, _d*invLen };
    }
};


struct AABB {
    Vec3 min;
    Vec3 max;

    static FORCE_INLINE AABB FromCenterExtents(const Vec3& c, const Vec3& e) noexcept {
        return { SUB(c, e), ADD(c, e) };
    }

    FORCE_INLINE Vec3 Center()  const noexcept { return MUL(ADD(min, max), 0.5f); }
    FORCE_INLINE Vec3 Extents() const noexcept { return MUL(SUB(max, min), 0.5f); }
    FORCE_INLINE Vec3 Size()    const noexcept { return SUB(max, min); }
    FORCE_INLINE float SurfaceArea() const noexcept {
        Vec3 d = Size();
        return 2.f * (d._x*d._y + d._y*d._z + d._z*d._x);
    }

    FORCE_INLINE bool Contains(const Vec3& p) const noexcept {
        return p._x >= min._x && p._x <= max._x &&
               p._y >= min._y && p._y <= max._y &&
               p._z >= min._z && p._z <= max._z;
    }

    FORCE_INLINE bool Overlaps(const AABB& o) const noexcept {
        return max._x >= o.min._x && min._x <= o.max._x &&
               max._y >= o.min._y && min._y <= o.max._y &&
               max._z >= o.min._z && min._z <= o.max._z;
    }

    FORCE_INLINE AABB Merge(const AABB& o) const noexcept {
        return { VMIN(min, o.min), VMAX(max, o.max) };
    }

    FORCE_INLINE AABB Expand(const Vec3& p) const noexcept {
        return { VMIN(min, p), VMAX(max, p) };
    }

    // Closest point on (or inside) the AABB to p
    FORCE_INLINE Vec3 ClosestPoint(const Vec3& p) const noexcept {
        return CLAMP(p, 0.f, 1.f); // placeholder — see full impl below
    }
    FORCE_INLINE Vec3 ClosestPointFull(const Vec3& p) const noexcept {
        return Vec3(Clampf(p._x, min._x, max._x),
                    Clampf(p._y, min._y, max._y),
                    Clampf(p._z, min._z, max._z));
    }
};

struct alignas(16) OBB {
    Vec3 center;        // world-space center
    Vec3 extents;       // half-sizes along local axes
    Quat orientation;   // local-to-world rotation

    FORCE_INLINE Vec3 Axis(int i) const noexcept {
        assert(i >= 0 && i < 3);
        Vec3 axes[3] = {
            QuatRotate(orientation, {1,0,0}),
            QuatRotate(orientation, {0,1,0}),
            QuatRotate(orientation, {0,0,1})
        };
        return axes[i];
    }

    // Project OBB onto a separating axis
    FORCE_INLINE float ProjectedRadius(const Vec3& axis) const noexcept {
        Vec3 a0 = Axis(0), a1 = Axis(1), a2 = Axis(2);
        return extents._x * std::abs(DOT(a0, axis))
             + extents._y * std::abs(DOT(a1, axis))
             + extents._z * std::abs(DOT(a2, axis));
    }
};

// SAT-based OBB vs OBB intersection test (15 separating axes)
inline bool OBBOverlaps(const OBB& a, const OBB& b) noexcept {
    Vec3 axesA[3] = { a.Axis(0), a.Axis(1), a.Axis(2) };
    Vec3 axesB[3] = { b.Axis(0), b.Axis(1), b.Axis(2) };
    Vec3 T = SUB(b.center, a.center);

    auto testAxis = [&](const Vec3& ax) {
        float ra = a.ProjectedRadius(ax);
        float rb = b.ProjectedRadius(ax);
        return std::abs(DOT(T, ax)) <= ra + rb;
    };

    for (int i = 0; i < 3; ++i) if (!testAxis(axesA[i])) return false;
    for (int i = 0; i < 3; ++i) if (!testAxis(axesB[i])) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Vec3 cx = CROSS(axesA[i], axesB[j]);
            float lenSq = LENGTH_SQ(cx);
            if (lenSq > EPSILON) if (!testAxis(MUL(cx, 1.f/std::sqrt(lenSq)))) return false;
        }
    return true;
}

struct Sphere { Vec3 center; float radius; };

FORCE_INLINE bool SphereOverlaps(const Sphere& a, const Sphere& b) noexcept {
    float rSum = a.radius + b.radius;
    return DISTANCE_SQ(a.center, b.center) <= rSum * rSum;
}

FORCE_INLINE bool SphereContainsAABB(const Sphere& s, const AABB& box) noexcept {
    // All 8 corners must be inside sphere
    float r2 = s.radius * s.radius;
    Vec3 corners[8] = {
        {box.min._x, box.min._y, box.min._z}, {box.max._x, box.min._y, box.min._z},
        {box.min._x, box.max._y, box.min._z}, {box.max._x, box.max._y, box.min._z},
        {box.min._x, box.min._y, box.max._z}, {box.max._x, box.min._y, box.max._z},
        {box.min._x, box.max._y, box.max._z}, {box.max._x, box.max._y, box.max._z}
    };
    for (auto& c : corners) if (DISTANCE_SQ(s.center, c) > r2) return false;
    return true;
}

FORCE_INLINE bool SphereOverlapsAABB(const Sphere& s, const AABB& box) noexcept {
    Vec3 cp = box.ClosestPointFull(s.center);
    return DISTANCE_SQ(s.center, cp) <= s.radius * s.radius;
}


struct Ray {
    Vec3 origin;
    Vec3 direction; // Must be normalised for distance queries

    FORCE_INLINE Vec3 At(float t) const noexcept {
        return ADD(origin, MUL(direction, t));
    }

    // Precomputed reciprocal direction for AABB tests (cache at call site)
    FORCE_INLINE Vec3 InvDir() const noexcept {
        return { 1.f/direction._x, 1.f/direction._y, 1.f/direction._z };
    }
};

FORCE_INLINE bool RayAABB(const Ray& ray, const AABB& box,
                           float& tNear, float& tFar) noexcept {
    __m128 ro  = _mm_load_ps(&ray.origin._x);
    __m128 rd  = _mm_load_ps(&ray.direction._x);
    __m128 bmi = _mm_load_ps(&box.min._x);
    __m128 bma = _mm_load_ps(&box.max._x);
    __m128 inv = _mm_div_ps(_mm_set1_ps(1.f), rd);
    __m128 t0  = _mm_mul_ps(_mm_sub_ps(bmi, ro), inv);
    __m128 t1  = _mm_mul_ps(_mm_sub_ps(bma, ro), inv);
    __m128 tlo = _mm_min_ps(t0, t1);
    __m128 thi = _mm_max_ps(t0, t1);
    __m128 tloYZ = _mm_shuffle_ps(tlo, tlo, _MM_SHUFFLE(3,3,2,1));
    __m128 tloX  = _mm_shuffle_ps(tlo, tlo, _MM_SHUFFLE(3,3,3,0));
    __m128 tnr   = _mm_max_ps(tloX, _mm_max_ps(
                        _mm_shuffle_ps(tloYZ, tloYZ, _MM_SHUFFLE(3,3,3,0)),
                        _mm_shuffle_ps(tloYZ, tloYZ, _MM_SHUFFLE(3,3,3,1))));
    __m128 thiYZ = _mm_shuffle_ps(thi, thi, _MM_SHUFFLE(3,3,2,1));
    __m128 thiX  = _mm_shuffle_ps(thi, thi, _MM_SHUFFLE(3,3,3,0));
    __m128 tfr   = _mm_min_ps(thiX, _mm_min_ps(
                        _mm_shuffle_ps(thiYZ, thiYZ, _MM_SHUFFLE(3,3,3,0)),
                        _mm_shuffle_ps(thiYZ, thiYZ, _MM_SHUFFLE(3,3,3,1))));
    tNear = _mm_cvtss_f32(tnr);
    tFar  = _mm_cvtss_f32(tfr);
    return tFar >= tNear && tFar >= 0.f;
}

FORCE_INLINE bool RayAABB(const Ray& ray, const AABB& box) noexcept {
    float tn, tf; return RayAABB(ray, box, tn, tf);
}

FORCE_INLINE bool RaySphere(const Ray& ray, const Sphere& sphere,
                             float& t) noexcept {
    Vec3  oc   = SUB(ray.origin, sphere.center);
    float a    = DOT(ray.direction, ray.direction);
    float h    = DOT(oc, ray.direction);
    float c    = DOT(oc, oc) - sphere.radius * sphere.radius;
    float disc = h * h - a * c;
    if (disc < 0.f) return false;
    float sqrtD = std::sqrt(disc);
    t = (-h - sqrtD) / a;
    if (t < EPSILON) {
        t = (-h + sqrtD) / a;
        if (t < EPSILON) return false;
    }
    return true;
}

FORCE_INLINE bool RayTriangle(const Ray& ray,
                               const Vec3& v0, const Vec3& v1, const Vec3& v2,
                               float& t, float& u, float& v) noexcept {
    Vec3 e1 = SUB(v1, v0);
    Vec3 e2 = SUB(v2, v0);
    Vec3 h  = CROSS(ray.direction, e2);
    float a = DOT(e1, h);
    if (std::abs(a) < EPSILON) return false;
    float f = 1.f / a;
    Vec3 s   = SUB(ray.origin, v0);
    u = f * DOT(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3 q = CROSS(s, e1);
    v = f * DOT(ray.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    t = f * DOT(e2, q);
    return t > EPSILON;
}

FORCE_INLINE bool RayPlane(const Ray& ray, const Plane& plane, float& t) noexcept {
    Vec3 n = { plane._a, plane._b, plane._c };
    float denom = DOT(ray.direction, n);
    if (std::abs(denom) < EPSILON) return false;
    t = -(DOT(ray.origin, n) + plane._d) / denom;
    return t >= 0.f;
}


struct Frustum {
    Plane planes[6]; // 0=left 1=right 2=bottom 3=top 4=near 5=far
};

// Extract frustum planes from a combined view-projection matrix (row-major)
FORCE_INLINE Frustum FrustumFromVP(const Mat4& vp) noexcept {
    // Gribb-Hartmann method
    auto Row = [&](int r) -> Vec4 {
        alignas(16) float f[4]; _mm_store_ps(f, vp.rows[r]);
        return { f[0], f[1], f[2], f[3] };
    };
    Vec4 r0 = Row(0), r1 = Row(1), r2 = Row(2), r3 = Row(3);

    auto MakePlane = [](float a, float b, float c, float d) -> Plane {
        Plane p = { a, b, c, d }; return p.Normalized();
    };
    Frustum f;
    f.planes[0] = MakePlane(r3._x + r0._x, r3._y + r0._y, r3._z + r0._z, r3._w + r0._w); // left
    f.planes[1] = MakePlane(r3._x - r0._x, r3._y - r0._y, r3._z - r0._z, r3._w - r0._w); // right
    f.planes[2] = MakePlane(r3._x + r1._x, r3._y + r1._y, r3._z + r1._z, r3._w + r1._w); // bottom
    f.planes[3] = MakePlane(r3._x - r1._x, r3._y - r1._y, r3._z - r1._z, r3._w - r1._w); // top
    f.planes[4] = MakePlane(r3._x + r2._x, r3._y + r2._y, r3._z + r2._z, r3._w + r2._w); // near
    f.planes[5] = MakePlane(r3._x - r2._x, r3._y - r2._y, r3._z - r2._z, r3._w - r2._w); // far
    return f;
}

FORCE_INLINE bool FrustumCullSphere(const Frustum& f,
                                     const Vec3& center, float radius) noexcept {
    __m128 cx = _mm_set1_ps(center._x);
    __m128 cy = _mm_set1_ps(center._y);
    __m128 cz = _mm_set1_ps(center._z);
    for (int i = 0; i < 6; ++i) {
        __m128 p  = _mm_load_ps(&f.planes[i]._a);
        __m128 nx = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0,0,0,0));
        __m128 ny = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1,1,1,1));
        __m128 nz = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,2,2,2));
        __m128 d  = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3,3,3,3));
        float dist = _mm_cvtss_f32(_mm_add_ps(d,
                         _mm_add_ps(_mm_mul_ps(nx, cx),
                         _mm_add_ps(_mm_mul_ps(ny, cy),
                                    _mm_mul_ps(nz, cz)))));
        if (dist < -radius) return true; // culled
    }
    return false;
}

FORCE_INLINE bool FrustumCullAABB(const Frustum& f, const AABB& box) noexcept {
    for (int i = 0; i < 6; ++i) {
        const Plane& p = f.planes[i];
        // Positive vertex (furthest along plane normal)
        Vec3 pv = {
            (p._a >= 0.f) ? box.max._x : box.min._x,
            (p._b >= 0.f) ? box.max._y : box.min._y,
            (p._c >= 0.f) ? box.max._z : box.min._z
        };
        if (p.SignedDistance(pv) < 0.f) return true; // fully outside this plane
    }
    return false;
}

struct alignas(16) ColorRGBA {
    float r, g, b, a;

    static FORCE_INLINE ColorRGBA White()       noexcept { return {1,1,1,1}; }
    static FORCE_INLINE ColorRGBA Black()       noexcept { return {0,0,0,1}; }
    static FORCE_INLINE ColorRGBA Transparent() noexcept { return {0,0,0,0}; }
    static FORCE_INLINE ColorRGBA Red()         noexcept { return {1,0,0,1}; }
    static FORCE_INLINE ColorRGBA Green()       noexcept { return {0,1,0,1}; }
    static FORCE_INLINE ColorRGBA Blue()        noexcept { return {0,0,1,1}; }

    // Premultiplied alpha
    FORCE_INLINE ColorRGBA Premultiplied() const noexcept {
        return { r*a, g*a, b*a, a };
    }

    // Encode as uint32_t RGBA8 (gamma-corrected, clamped)
    FORCE_INLINE uint32_t ToRGBA8() const noexcept {
        auto ToU8 = [](float v) -> uint32_t {
            return static_cast<uint32_t>(Clampf(v, 0.f, 1.f) * 255.f + 0.5f);
        };
        return (ToU8(r) << 24) | (ToU8(g) << 16) | (ToU8(b) << 8) | ToU8(a);
    }

    static FORCE_INLINE ColorRGBA FromRGBA8(uint32_t rgba) noexcept {
        constexpr float inv = 1.f / 255.f;
        return {
            ((rgba >> 24) & 0xFF) * inv,
            ((rgba >> 16) & 0xFF) * inv,
            ((rgba >>  8) & 0xFF) * inv,
            ((rgba      ) & 0xFF) * inv
        };
    }

    FORCE_INLINE ColorRGBA Lerp(const ColorRGBA& to, float t) const noexcept {
        return {
            r + t*(to.r - r), g + t*(to.g - g),
            b + t*(to.b - b), a + t*(to.a - a)
        };
    }
};

struct alignas(32) Mat4SoA {
    float m[4][4][8]; // m[row][col][lane]

    Mat4SoA() noexcept { std::memset(m, 0, sizeof(m)); }

    FORCE_INLINE void SetIdentity(int lane) noexcept {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m[r][c][lane] = (r == c) ? 1.f : 0.f;
    }

    FORCE_INLINE void Set(int lane, const Mat4& mat) noexcept {
        for (int r = 0; r < 4; ++r) {
            alignas(16) float tmp[4]; _mm_store_ps(tmp, mat.rows[r]);
            for (int c = 0; c < 4; ++c) m[r][c][lane] = tmp[c];
        }
    }

    FORCE_INLINE Mat4 Get(int lane) const noexcept {
        Mat4 out;
        for (int r = 0; r < 4; ++r)
            out.rows[r] = _mm_setr_ps(m[r][0][lane], m[r][1][lane],
                                      m[r][2][lane], m[r][3][lane]);
        return out;
    }
};

/// Transform 8 Vec4s by 8 matrices simultaneously (one lane per pair)
FORCE_INLINE void TransformVec4SoA(const Mat4SoA& matrices,
                                    const Vec4SoA& vectors,
                                    Vec4SoA& RESTRICT out) noexcept {
#if FLAME_AVX
    for (int r = 0; r < 4; ++r) {
        __m256 acc = _mm256_setzero_ps();
        for (int c = 0; c < 4; ++c) {
            __m256 mc = _mm256_load_ps(matrices.m[r][c]);
            __m256 vc = (c == 0) ? _mm256_load_ps(vectors.x)
                      : (c == 1) ? _mm256_load_ps(vectors.y)
                      : (c == 2) ? _mm256_load_ps(vectors.z)
                                 : _mm256_load_ps(vectors.w);
            acc = _mm256_fmadd_ps(mc, vc, acc);
        }
        float* dst = (r == 0) ? out.x : (r == 1) ? out.y : (r == 2) ? out.z : out.w;
        _mm256_store_ps(dst, acc);
    }
#else
    // SSE fallback: process lower 4 then upper 4 lanes
    for (int half = 0; half < 2; ++half) {
        int off = half * 4;
        for (int r = 0; r < 4; ++r) {
            __m128 acc = _mm_setzero_ps();
            for (int c = 0; c < 4; ++c) {
                __m128 mc = _mm_load_ps(matrices.m[r][c] + off);
                const float* vs = (c == 0) ? vectors.x : (c == 1) ? vectors.y
                                : (c == 2) ? vectors.z : vectors.w;
                __m128 vc = _mm_load_ps(vs + off);
                acc = _mm_add_ps(acc, _mm_mul_ps(mc, vc));
            }
            float* dst = (r == 0) ? out.x : (r == 1) ? out.y : (r == 2) ? out.z : out.w;
            _mm_store_ps(dst + off, acc);
        }
    }
#endif
}

/// Prefetch a single cache-line (typically 64 bytes) into L1
template <typename T>
FORCE_INLINE void PrefetchL1(const T* ptr) noexcept { FLAME_PREFETCH_L1(ptr); }

/// Prefetch into L2
template <typename T>
FORCE_INLINE void PrefetchL2(const T* ptr) noexcept { FLAME_PREFETCH_L2(ptr); }

/// Non-temporal prefetch (streaming read, bypasses cache)
template <typename T>
FORCE_INLINE void PrefetchNT(const T* ptr) noexcept { FLAME_PREFETCH_NT(ptr); }

/// Prefetch a full struct spanning multiple cache lines
template <typename T>
FORCE_INLINE void PrefetchStruct(const T* ptr) noexcept {
    constexpr size_t CL = 64;
    const char* p = reinterpret_cast<const char*>(ptr);
    for (size_t off = 0; off < sizeof(T); off += CL)
        FLAME_PREFETCH_L1(p + off);
}

/// Prefetch a contiguous array ahead by `lookahead` elements
template <typename T>
FORCE_INLINE void PrefetchArray(const T* base, size_t idx,
                                 size_t lookahead = 4) noexcept {
    FLAME_PREFETCH_L1(base + idx + lookahead);
}

// ===========================================================================
// Operator overloads
// ===========================================================================

// --- Vec2 ---
FORCE_INLINE Vec2 operator+(const Vec2& a, const Vec2& b) noexcept { return ADD(a, b); }
FORCE_INLINE Vec2 operator-(const Vec2& a, const Vec2& b) noexcept { return SUB(a, b); }
FORCE_INLINE Vec2 operator*(const Vec2& a, float s)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec2 operator*(float s, const Vec2& a)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec2 operator/(const Vec2& a, float s)       noexcept { return DIV(a, s); }
FORCE_INLINE Vec2 operator-(const Vec2& a)                noexcept { return NEG(a); }
FORCE_INLINE Vec2& operator+=(Vec2& a, const Vec2& b)     noexcept { a = ADD(a, b); return a; }
FORCE_INLINE Vec2& operator-=(Vec2& a, const Vec2& b)     noexcept { a = SUB(a, b); return a; }
FORCE_INLINE Vec2& operator*=(Vec2& a, float s)           noexcept { a = MUL(a, s); return a; }

// --- Vec3 ---
FORCE_INLINE Vec3 operator+(const Vec3& a, const Vec3& b) noexcept { return ADD(a, b); }
FORCE_INLINE Vec3 operator-(const Vec3& a, const Vec3& b) noexcept { return SUB(a, b); }
FORCE_INLINE Vec3 operator*(const Vec3& a, float s)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec3 operator*(float s, const Vec3& a)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec3 operator/(const Vec3& a, float s)       noexcept { return DIV(a, s); }
FORCE_INLINE Vec3 operator-(const Vec3& a)                noexcept { return NEG(a); }
FORCE_INLINE Vec3& operator+=(Vec3& a, const Vec3& b)     noexcept { a = ADD(a, b); return a; }
FORCE_INLINE Vec3& operator-=(Vec3& a, const Vec3& b)     noexcept { a = SUB(a, b); return a; }
FORCE_INLINE Vec3& operator*=(Vec3& a, float s)           noexcept { a = MUL(a, s); return a; }

// --- Vec4 ---
FORCE_INLINE Vec4 operator+(const Vec4& a, const Vec4& b) noexcept { return ADD(a, b); }
FORCE_INLINE Vec4 operator-(const Vec4& a, const Vec4& b) noexcept { return SUB(a, b); }
FORCE_INLINE Vec4 operator*(const Vec4& a, float s)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec4 operator*(float s, const Vec4& a)       noexcept { return MUL(a, s); }
FORCE_INLINE Vec4& operator+=(Vec4& a, const Vec4& b)     noexcept { a = ADD(a, b); return a; }
FORCE_INLINE Vec4& operator-=(Vec4& a, const Vec4& b)     noexcept { a = SUB(a, b); return a; }

// --- Mat4 ---
FORCE_INLINE Mat4 operator*(const Mat4& a, const Mat4& b) noexcept { return MUL(a, b); }
FORCE_INLINE Vec4 operator*(const Mat4& m, const Vec4& v) noexcept { return TransformVec4(m, v); }
FORCE_INLINE Mat4& operator*=(Mat4& a, const Mat4& b)     noexcept { a = MUL(a, b); return a; }

// --- Quat ---
FORCE_INLINE Quat operator*(const Quat& a, const Quat& b) noexcept { return QuatMul(a, b); }
FORCE_INLINE Quat& operator*=(Quat& a, const Quat& b)     noexcept { a = QuatMul(a, b); return a; }

// ===========================================================================
// CPU feature detection
// ===========================================================================
inline bool CPU_HasSSE4() noexcept {
#if defined(_MSC_VER)
    int info[4]; __cpuid(info, 1);
    return (info[2] & (1 << 19)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("sse4.1");
#else
    return false;
#endif
}

inline bool CPU_HasAVX2() noexcept {
#if defined(_MSC_VER)
    int info[4]; __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

inline bool CPU_HasFMA() noexcept {
#if defined(_MSC_VER)
    int info[4]; __cpuid(info, 1);
    return (info[2] & (1 << 12)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

} // namespace flame