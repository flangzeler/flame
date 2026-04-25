 /*
flame.h by FLANGZELER

DATE: 04-25-2026 | RELEASE: 1.0.2

LICENSE: This code is released under the MIT License. See LICENSE file for details.

flame (FLANGZELER's MATH ENGINE) is a SIMD based math library made to provide high-performance mathematical operations whether on CPU or GPU.
It can be used for various mathematical computations, including vector and matrix operations, quaternion manipulations, and more.

This library is made to give the developers ease in making high-performance applications, such as games, simulations, and scientific computing, by leveraging the power of SIMD instructions.

*/
#pragma once

#include <immintrin.h>
#include <concepts>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cassert>
#include <memory>

#if defined(__GNUC__) || defined(__clang__)
#  define RESTRICT      __restrict__
#  define FORCE_INLINE  __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#  define RESTRICT      __restrict
#  define FORCE_INLINE  __forceinline
#else
#  define RESTRICT
#  define FORCE_INLINE  inline
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

    constexpr float PI = 3.14159265358979323846f;
    constexpr float TWO_PI = 6.28318530717958647692f;
    constexpr float HALF_PI = 1.57079632679489661923f;
    constexpr float INV_PI = 0.31830988618379067154f;
    constexpr float DEG2RAD = PI / 180.0f;
    constexpr float RAD2DEG = 180.0f / PI;
    constexpr float EPSILON = 1e-6f;
    constexpr float F32_MAX = FLT_MAX;

    FORCE_INLINE float ToRadians(float d) { return d * DEG2RAD; }
    FORCE_INLINE float ToDegrees(float r) { return r * RAD2DEG; }
    FORCE_INLINE float Clampf(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    FORCE_INLINE float LaneF(const __m128 r, int i) {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, r);
        return tmp[i];
    }

    struct alignas(ALIGNMENT) Vec2 {
        float _x, _y, _z, _w;   
        FORCE_INLINE Vec2(float x = 0.f, float y = 0.f)
            : _x(x), _y(y), _z(0.f), _w(0.f) {
        }
    };

    struct alignas(ALIGNMENT) Vec3 {
        float _x, _y, _z, _w;   
        FORCE_INLINE Vec3(float x = 0.f, float y = 0.f, float z = 0.f)
            : _x(x), _y(y), _z(z), _w(0.f) {
        }
    };

    struct alignas(ALIGNMENT) Vec4 {
        float _x, _y, _z, _w;
        FORCE_INLINE Vec4(float x = 0.f, float y = 0.f,
            float z = 0.f, float w = 1.f)
            : _x(x), _y(y), _z(z), _w(w) {
        }
    };

    struct alignas(ALIGNMENT) Quat {
        float _x, _y, _z, _w;
        FORCE_INLINE Quat(float x = 0.f, float y = 0.f,
            float z = 0.f, float w = 1.f)
            : _x(x), _y(y), _z(z), _w(w) {
        }
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

        Vec4SoA() {
            for (int i = 0; i < 8; ++i) x[i] = y[i] = z[i] = 0.f, w[i] = 1.f;
        }

        FORCE_INLINE void Set(int lane, const Vec4& v) {
            assert(lane >= 0 && lane < 8);
            x[lane] = v._x; y[lane] = v._y;
            z[lane] = v._z; w[lane] = v._w;
        }

        FORCE_INLINE Vec4 Get(int lane) const {
            assert(lane >= 0 && lane < 8);
            return { x[lane], y[lane], z[lane], w[lane] };
        }
    };

    FORCE_INLINE Vec4SoA AddSoA(const Vec4SoA& a, const Vec4SoA& b) {
        Vec4SoA out;
#if FLAME_AVX
        _mm256_store_ps(out.x, _mm256_add_ps(_mm256_load_ps(a.x), _mm256_load_ps(b.x)));
        _mm256_store_ps(out.y, _mm256_add_ps(_mm256_load_ps(a.y), _mm256_load_ps(b.y)));
        _mm256_store_ps(out.z, _mm256_add_ps(_mm256_load_ps(a.z), _mm256_load_ps(b.z)));
        _mm256_store_ps(out.w, _mm256_add_ps(_mm256_load_ps(a.w), _mm256_load_ps(b.w)));
#else
        _mm_store_ps(out.x, _mm_add_ps(_mm_load_ps(a.x), _mm_load_ps(b.x)));
        _mm_store_ps(out.x + 4, _mm_add_ps(_mm_load_ps(a.x + 4), _mm_load_ps(b.x + 4)));
        _mm_store_ps(out.y, _mm_add_ps(_mm_load_ps(a.y), _mm_load_ps(b.y)));
        _mm_store_ps(out.y + 4, _mm_add_ps(_mm_load_ps(a.y + 4), _mm_load_ps(b.y + 4)));
        _mm_store_ps(out.z, _mm_add_ps(_mm_load_ps(a.z), _mm_load_ps(b.z)));
        _mm_store_ps(out.z + 4, _mm_add_ps(_mm_load_ps(a.z + 4), _mm_load_ps(b.z + 4)));
        _mm_store_ps(out.w, _mm_add_ps(_mm_load_ps(a.w), _mm_load_ps(b.w)));
        _mm_store_ps(out.w + 4, _mm_add_ps(_mm_load_ps(a.w + 4), _mm_load_ps(b.w + 4)));
#endif
        return out;
    }

    FORCE_INLINE Vec4SoA MulSoA(const Vec4SoA& a, float s) {
        Vec4SoA out;
#if FLAME_AVX
        __m256 vs = _mm256_set1_ps(s);
        _mm256_store_ps(out.x, _mm256_mul_ps(_mm256_load_ps(a.x), vs));
        _mm256_store_ps(out.y, _mm256_mul_ps(_mm256_load_ps(a.y), vs));
        _mm256_store_ps(out.z, _mm256_mul_ps(_mm256_load_ps(a.z), vs));
        _mm256_store_ps(out.w, _mm256_mul_ps(_mm256_load_ps(a.w), vs));
#else
        __m128 vs = _mm_set1_ps(s);
        _mm_store_ps(out.x, _mm_mul_ps(_mm_load_ps(a.x), vs));
        _mm_store_ps(out.x + 4, _mm_mul_ps(_mm_load_ps(a.x + 4), vs));
        _mm_store_ps(out.y, _mm_mul_ps(_mm_load_ps(a.y), vs));
        _mm_store_ps(out.y + 4, _mm_mul_ps(_mm_load_ps(a.y + 4), vs));
        _mm_store_ps(out.z, _mm_mul_ps(_mm_load_ps(a.z), vs));
        _mm_store_ps(out.z + 4, _mm_mul_ps(_mm_load_ps(a.z + 4), vs));
        _mm_store_ps(out.w, _mm_mul_ps(_mm_load_ps(a.w), vs));
        _mm_store_ps(out.w + 4, _mm_mul_ps(_mm_load_ps(a.w + 4), vs));
#endif
        return out;
    }

    FORCE_INLINE void DotSoA(const Vec4SoA& a, const Vec4SoA& b,
        float* RESTRICT out8) {
#if FLAME_AVX
        __m256 ax = _mm256_load_ps(a.x), bx = _mm256_load_ps(b.x);
        __m256 ay = _mm256_load_ps(a.y), by = _mm256_load_ps(b.y);
        __m256 az = _mm256_load_ps(a.z), bz = _mm256_load_ps(b.z);
        __m256 d = _mm256_fmadd_ps(ax, bx,
            _mm256_fmadd_ps(ay, by,
                _mm256_mul_ps(az, bz)));
        _mm256_store_ps(out8, d);
#else
        __m128 ax0 = _mm_load_ps(a.x), bx0 = _mm_load_ps(b.x);
        __m128 ax1 = _mm_load_ps(a.x + 4), bx1 = _mm_load_ps(b.x + 4);
        __m128 ay0 = _mm_load_ps(a.y), by0 = _mm_load_ps(b.y);
        __m128 ay1 = _mm_load_ps(a.y + 4), by1 = _mm_load_ps(b.y + 4);
        __m128 az0 = _mm_load_ps(a.z), bz0 = _mm_load_ps(b.z);
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
    FORCE_INLINE T ADD(const T& a, const T& b) {
        T out;
        _mm_store_ps(&out._x, _mm_add_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T SUB(const T& a, const T& b) {
        T out;
        _mm_store_ps(&out._x, _mm_sub_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T MUL(const T& a, float s) {
        T out;
        _mm_store_ps(&out._x, _mm_mul_ps(_mm_load_ps(&a._x), _mm_set1_ps(s)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T DIV(const T& a, float s) {
        T out;
        _mm_store_ps(&out._x, _mm_div_ps(_mm_load_ps(&a._x), _mm_set1_ps(s)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T LERP(const T& a, const T& b, float t) {
        T out;
        __m128 vt = _mm_set1_ps(t);
        __m128 va = _mm_load_ps(&a._x);
        __m128 vb = _mm_load_ps(&b._x);
        _mm_store_ps(&out._x, _mm_add_ps(va, _mm_mul_ps(vt, _mm_sub_ps(vb, va))));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T CLAMP(const T& v, float lo, float hi) {
        T out;
        _mm_store_ps(&out._x,
            _mm_min_ps(_mm_max_ps(_mm_load_ps(&v._x), _mm_set1_ps(lo)),
                _mm_set1_ps(hi)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T VMIN(const T& a, const T& b) {
        T out;
        _mm_store_ps(&out._x, _mm_min_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T VMAX(const T& a, const T& b) {
        T out;
        _mm_store_ps(&out._x, _mm_max_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE T NEG(const T& a) {
        T out;
        _mm_store_ps(&out._x,
            _mm_sub_ps(_mm_setzero_ps(), _mm_load_ps(&a._x)));
        return out;
    }

    template <vec_type T>
    FORCE_INLINE float DOT(const T& a, const T& b) {
        return _mm_cvtss_f32(
            _mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x), 0x71));
    }

    FORCE_INLINE float DOT4(const Vec4& a, const Vec4& b) {
        return _mm_cvtss_f32(
            _mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x), 0xF1));
    }

    template <vec_type T>
    FORCE_INLINE float LENGTH(const T& a) {
        __m128 reg = _mm_load_ps(&a._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0x71);
        return _mm_cvtss_f32(_mm_sqrt_ps(lsq));
    }

    FORCE_INLINE float LENGTH4(const Vec4& a) {
        __m128 reg = _mm_load_ps(&a._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0xF1);
        return _mm_cvtss_f32(_mm_sqrt_ps(lsq));
    }

    template <vec_type T>
    FORCE_INLINE T NORMALIZE(const T& a) {
        __m128 reg = _mm_load_ps(&a._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0x7F);     
        __m128 est = _mm_rsqrt_ps(lsq);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 inv = _mm_mul_ps(half,
            _mm_mul_ps(est,
                _mm_sub_ps(three, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
        T out;
        _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
        return out;
    }

    FORCE_INLINE Vec4 NORMALIZE4(const Vec4& a) {
        __m128 reg = _mm_load_ps(&a._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0xFF);
        __m128 est = _mm_rsqrt_ps(lsq);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three = _mm_set1_ps(3.0f);
        __m128 inv = _mm_mul_ps(half,
            _mm_mul_ps(est,
                _mm_sub_ps(three, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
        Vec4 out;
        _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
        return out;
    }

    FORCE_INLINE Vec3 CROSS(const Vec3& a, const Vec3& b) {
        __m128 rA = _mm_load_ps(&a._x);
        __m128 rB = _mm_load_ps(&b._x);
        __m128 t0 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 0, 2, 1));  
        __m128 t1 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 1, 0, 2));  
        __m128 t2 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 1, 0, 2));  
        __m128 t3 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 0, 2, 1));  
        Vec3 out;
        _mm_store_ps(&out._x, _mm_sub_ps(_mm_mul_ps(t0, t1), _mm_mul_ps(t2, t3)));
        return out;
    }

    FORCE_INLINE Vec3 REFLECT(const Vec3& v, const Vec3& n) {
        float d2 = 2.f * DOT(v, n);
        return SUB(v, MUL(n, d2));
    }

    FORCE_INLINE Vec3 REFRACT(const Vec3& v, const Vec3& n, float eta) {
        float cosI = -DOT(v, n);
        float sinT2 = eta * eta * (1.f - cosI * cosI);
        if (sinT2 > 1.f) return Vec3(0, 0, 0);    
        float cosT = std::sqrt(1.f - sinT2);
        return ADD(MUL(v, eta), MUL(n, eta * cosI - cosT));
    }

    template <vec_type T>
    FORCE_INLINE float DISTANCE(const T& a, const T& b) {
        return LENGTH(SUB(a, b));
    }

    FORCE_INLINE Quat QuatIdentity() { return { 0.f, 0.f, 0.f, 1.f }; }

    FORCE_INLINE Quat QuatNormalize(const Quat& q) {
        __m128 reg = _mm_load_ps(&q._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0xFF);
        __m128 est = _mm_rsqrt_ps(lsq);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 thr = _mm_set1_ps(3.0f);
        __m128 inv = _mm_mul_ps(half,
            _mm_mul_ps(est,
                _mm_sub_ps(thr, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
        Quat out;
        _mm_store_ps(&out._x, _mm_mul_ps(reg, inv));
        return out;
    }

    FORCE_INLINE Quat QuatConjugate(const Quat& q) {
        static const __m128 SIGN_MASK =
            _mm_set_ps(0.f, -0.f, -0.f, -0.f);       
        Quat out;
        _mm_store_ps(&out._x, _mm_xor_ps(_mm_load_ps(&q._x), SIGN_MASK));
        return out;
    }

    FORCE_INLINE Quat QuatMul(const Quat& a, const Quat& b) {
        float ax = a._x, ay = a._y, az = a._z, aw = a._w;
        float bx = b._x, by = b._y, bz = b._z, bw = b._w;
        return {
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz
        };
    }

    FORCE_INLINE Quat QuatFromAxisAngle(const Vec3& axis, float angle) {
        float half = angle * 0.5f;
        float s = std::sin(half);
        float c = std::cos(half);
        return { axis._x * s, axis._y * s, axis._z * s, c };
    }

    FORCE_INLINE Vec3 QuatRotate(const Quat& q, const Vec3& v) {
        Vec3 qv = { q._x, q._y, q._z };
        Vec3 t = MUL(CROSS(qv, v), 2.f);
        return ADD(ADD(v, MUL(t, q._w)), CROSS(qv, t));
    }

    FORCE_INLINE Quat QuatSlerp(const Quat& a, const Quat& b, float t) {
        float dot = a._x * b._x + a._y * b._y + a._z * b._z + a._w * b._w;

        Quat bAdj = b;
        if (dot < 0.f) {
            dot = -dot;
            bAdj = { -b._x, -b._y, -b._z, -b._w };
        }

        if (dot > 0.9995f) {
            __m128 va = _mm_load_ps(&a._x);
            __m128 vb = _mm_load_ps(&bAdj._x);
            __m128 vt = _mm_set1_ps(t);
            __m128 res = _mm_add_ps(va, _mm_mul_ps(vt, _mm_sub_ps(vb, va)));
            __m128 lsq = _mm_dp_ps(res, res, 0xFF);
            __m128 est = _mm_rsqrt_ps(lsq);
            __m128 half_v = _mm_set1_ps(0.5f);
            __m128 thr_v = _mm_set1_ps(3.0f);
            __m128 inv = _mm_mul_ps(half_v,
                _mm_mul_ps(est,
                    _mm_sub_ps(thr_v, _mm_mul_ps(lsq, _mm_mul_ps(est, est)))));
            Quat out;
            _mm_store_ps(&out._x, _mm_mul_ps(res, inv));
            return out;
        }

        float theta0 = std::acos(dot);
        float sinTheta0 = std::sin(theta0);
        float theta = theta0 * t;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        float s0 = cosTheta - dot * sinTheta / sinTheta0;
        float s1 = sinTheta / sinTheta0;

        __m128 va = _mm_load_ps(&a._x);
        __m128 vb = _mm_load_ps(&bAdj._x);
        __m128 res = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(s0), va),
            _mm_mul_ps(_mm_set1_ps(s1), vb));
        Quat out;
        _mm_store_ps(&out._x, res);
        return out;
    }

    FORCE_INLINE void QuatToMat4(const Quat& q, struct Mat4& out);

    struct alignas(ALIGNMENT) Mat4 {
        __m128 rows[4];

        static FORCE_INLINE Mat4 Identity() {
            Mat4 m;
            m.rows[0] = _mm_setr_ps(1, 0, 0, 0);
            m.rows[1] = _mm_setr_ps(0, 1, 0, 0);
            m.rows[2] = _mm_setr_ps(0, 0, 1, 0);
            m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
            return m;
        }

        static FORCE_INLINE Mat4 Zero() {
            Mat4 m;
            m.rows[0] = m.rows[1] = m.rows[2] = m.rows[3] = _mm_setzero_ps();
            return m;
        }

        FORCE_INLINE float Get(int row, int col) const {
            return LaneF(rows[row], col);
        }
        FORCE_INLINE void Set(int row, int col, float v) {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, rows[row]);
            tmp[col] = v;
            rows[row] = _mm_load_ps(tmp);
        }
    };

    FORCE_INLINE Mat4 MUL(const Mat4& a, const Mat4& b) {
        Mat4 out;
        for (int i = 0; i < 4; ++i) {
            __m128 row = a.rows[i];
            __m128 res = _mm_mul_ps(_mm_shuffle_ps(row, row, _MM_SHUFFLE(0, 0, 0, 0)), b.rows[0]);
            res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row, row, _MM_SHUFFLE(1, 1, 1, 1)), b.rows[1]));
            res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row, row, _MM_SHUFFLE(2, 2, 2, 2)), b.rows[2]));
            res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(row, row, _MM_SHUFFLE(3, 3, 3, 3)), b.rows[3]));
            out.rows[i] = res;
        }
        return out;
    }

    FORCE_INLINE Vec4 TransformVec4(const Mat4& m, const Vec4& v) {
        __m128 vv = _mm_load_ps(&v._x);
        __m128 res = _mm_mul_ps(_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(0, 0, 0, 0)), m.rows[0]);
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(1, 1, 1, 1)), m.rows[1]));
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(2, 2, 2, 2)), m.rows[2]));
        res = _mm_add_ps(res, _mm_mul_ps(_mm_shuffle_ps(vv, vv, _MM_SHUFFLE(3, 3, 3, 3)), m.rows[3]));
        Vec4 out;
        _mm_store_ps(&out._x, res);
        return out;
    }

    FORCE_INLINE Vec3 TransformPoint(const Mat4& m, const Vec3& p) {
        Vec4 v4 = { p._x, p._y, p._z, 1.f };
        Vec4 r = TransformVec4(m, v4);
        float invW = 1.f / r._w;
        return { r._x * invW, r._y * invW, r._z * invW };
    }

    FORCE_INLINE Vec3 TransformDir(const Mat4& m, const Vec3& d) {
        Vec4 v4 = { d._x, d._y, d._z, 0.f };
        Vec4 r = TransformVec4(m, v4);
        return { r._x, r._y, r._z };
    }

    FORCE_INLINE Mat4 Transpose(const Mat4& m) {
        Mat4 out = m;
        _MM_TRANSPOSE4_PS(out.rows[0], out.rows[1], out.rows[2], out.rows[3]);
        return out;
    }

    FORCE_INLINE Mat4 Inverse(const Mat4& src) {
        Mat4 m = Transpose(src);

        __m128 row0 = m.rows[0], row1 = m.rows[1];
        __m128 row2 = m.rows[2], row3 = m.rows[3];

        __m128 tmp1, minor0, minor1, minor2, minor3;
        __m128 det;

        tmp1 = _mm_mul_ps(row2, row3);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        minor0 = _mm_mul_ps(row1, tmp1);
        minor1 = _mm_mul_ps(row0, tmp1);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
        minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
        minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

        tmp1 = _mm_mul_ps(row1, row2);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
        minor3 = _mm_mul_ps(row0, tmp1);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
        minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
        minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

        tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        row2 = _mm_shuffle_ps(row2, row2, 0x4E);
        minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
        minor2 = _mm_mul_ps(row0, tmp1);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
        minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
        minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

        tmp1 = _mm_mul_ps(row0, row1);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
        minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
        minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

        tmp1 = _mm_mul_ps(row0, row3);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
        minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
        minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

        tmp1 = _mm_mul_ps(row0, row2);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
        minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
        minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

        det = _mm_mul_ps(row0, minor0);
        det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
        det = _mm_add_ps(_mm_shuffle_ps(det, det, 0xB1), det);
        tmp1 = _mm_rcp_ps(det);
        det = _mm_sub_ps(_mm_add_ps(tmp1, tmp1),
            _mm_mul_ps(det, _mm_mul_ps(tmp1, tmp1)));

        Mat4 out;
        out.rows[0] = _mm_mul_ps(det, minor0);
        out.rows[1] = _mm_mul_ps(det, minor1);
        out.rows[2] = _mm_mul_ps(det, minor2);
        out.rows[3] = _mm_mul_ps(det, minor3);
        return out;
    }

    FORCE_INLINE float Determinant(const Mat4& m) {
        float a = m.Get(0, 0), b = m.Get(0, 1), c = m.Get(0, 2), d = m.Get(0, 3);
        float m10 = m.Get(1, 0), m11 = m.Get(1, 1), m12 = m.Get(1, 2), m13 = m.Get(1, 3);
        float m20 = m.Get(2, 0), m21 = m.Get(2, 1), m22 = m.Get(2, 2), m23 = m.Get(2, 3);
        float m30 = m.Get(3, 0), m31 = m.Get(3, 1), m32 = m.Get(3, 2), m33 = m.Get(3, 3);

        float C0 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
        float C1 = m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30);
        float C2 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30);
        float C3 = m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30);

        return a * C0 - b * C1 + c * C2 - d * C3;
    }

    FORCE_INLINE Mat4 Translate(const Vec3& v) {
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(1, 0, 0, v._x);
        m.rows[1] = _mm_setr_ps(0, 1, 0, v._y);
        m.rows[2] = _mm_setr_ps(0, 0, 1, v._z);
        return m;
    }

    FORCE_INLINE Mat4 Scale(const Vec3& v) {
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(v._x, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, v._y, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, v._z, 0);
        return m;
    }

    FORCE_INLINE Mat4 Scale(float s) {
        return Scale({ s, s, s });
    }

    FORCE_INLINE Mat4 RotateX(float angle) {
        float s = std::sin(angle), c = std::cos(angle);
        Mat4 m = Mat4::Identity();
        m.rows[1] = _mm_setr_ps(0, c, -s, 0);
        m.rows[2] = _mm_setr_ps(0, s, c, 0);
        return m;
    }

    FORCE_INLINE Mat4 RotateY(float angle) {
        float s = std::sin(angle), c = std::cos(angle);
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(c, 0, s, 0);
        m.rows[2] = _mm_setr_ps(-s, 0, c, 0);
        return m;
    }

    FORCE_INLINE Mat4 RotateZ(float angle) {
        float s = std::sin(angle), c = std::cos(angle);
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(c, -s, 0, 0);
        m.rows[1] = _mm_setr_ps(s, c, 0, 0);
        return m;
    }

    FORCE_INLINE Mat4 RotateAxisAngle(const Vec3& axis, float angle) {
        Vec3 u = NORMALIZE(axis);
        float s = std::sin(angle), c = std::cos(angle), ic = 1.f - c;
        float x = u._x, y = u._y, z = u._z;
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(c + x * x * ic, x * y * ic - z * s, x * z * ic + y * s, 0);
        m.rows[1] = _mm_setr_ps(y * x * ic + z * s, c + y * y * ic, y * z * ic - x * s, 0);
        m.rows[2] = _mm_setr_ps(z * x * ic - y * s, z * y * ic + x * s, c + z * z * ic, 0);
        return m;
    }

    FORCE_INLINE void QuatToMat4(const Quat& q, Mat4& out) {
        float x = q._x, y = q._y, z = q._z, w = q._w;
        float x2 = x + x, y2 = y + y, z2 = z + z;
        float xx = x * x2, xy = x * y2, xz = x * z2;
        float yy = y * y2, yz = y * z2, zz = z * z2;
        float wx = w * x2, wy = w * y2, wz = w * z2;

        out = Mat4::Identity();
        out.rows[0] = _mm_setr_ps(1 - (yy + zz), xy - wz, xz + wy, 0);
        out.rows[1] = _mm_setr_ps(xy + wz, 1 - (xx + zz), yz - wx, 0);
        out.rows[2] = _mm_setr_ps(xz - wy, yz + wx, 1 - (xx + yy), 0);
    }

    FORCE_INLINE Mat4 QuatToMat4(const Quat& q) {
        Mat4 out;
        QuatToMat4(q, out);
        return out;
    }

    FORCE_INLINE Mat4 TRS(const Vec3& t, const Quat& r, const Vec3& s) {
        return MUL(MUL(Translate(t), QuatToMat4(r)), Scale(s));
    }

    FORCE_INLINE Mat4 Perspective(float fovDeg, float aspect,
        float nearP, float farP) {
        float f = 1.f / std::tan(ToRadians(fovDeg) * 0.5f);
        float range = farP - nearP;
        Mat4 m;
        m.rows[0] = _mm_setr_ps(f / aspect, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, f, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, farP / range, -(farP * nearP) / range);
        m.rows[3] = _mm_setr_ps(0, 0, 1, 0);
        return m;
    }

    FORCE_INLINE Mat4 Orthographic(float left, float right,
        float bottom, float top,
        float nearP, float farP) {
        float rml = right - left, rpl = right + left;
        float tmb = top - bottom, tpb = top + bottom;
        float fmn = farP - nearP, fpn = farP + nearP;
        Mat4 m;
        m.rows[0] = _mm_setr_ps(2 / rml, 0, 0, -rpl / rml);
        m.rows[1] = _mm_setr_ps(0, 2 / tmb, 0, -tpb / tmb);
        m.rows[2] = _mm_setr_ps(0, 0, 2 / fmn, -fpn / fmn);
        m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
        return m;
    }

    FORCE_INLINE Mat4 LookAt(const Vec3& eye, const Vec3& target,
        const Vec3& up) {
        Vec3 z = NORMALIZE(SUB(target, eye));
        Vec3 x = NORMALIZE(CROSS(z, up));          
        Vec3 y = CROSS(x, z);
        Mat4 m;
        m.rows[0] = _mm_setr_ps(x._x, x._y, x._z, -DOT(x, eye));
        m.rows[1] = _mm_setr_ps(y._x, y._y, y._z, -DOT(y, eye));
        m.rows[2] = _mm_setr_ps(-z._x, -z._y, -z._z, DOT(z, eye));
        m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
        return m;
    }

    struct Ray {
        Vec3 origin;
        Vec3 direction;      

        FORCE_INLINE Vec3 At(float t) const {
            return ADD(origin, MUL(direction, t));
        }
    };

    struct AABB {
        Vec3 min;    
        Vec3 max;    

        static FORCE_INLINE AABB FromCenterExtents(const Vec3& c, const Vec3& e) {
            return { SUB(c,e), ADD(c,e) };
        }

        FORCE_INLINE Vec3 Center()  const { return MUL(ADD(min, max), 0.5f); }
        FORCE_INLINE Vec3 Extents() const { return MUL(SUB(max, min), 0.5f); }

        FORCE_INLINE bool Contains(const Vec3& p) const {
            return p._x >= min._x && p._x <= max._x &&
                p._y >= min._y && p._y <= max._y &&
                p._z >= min._z && p._z <= max._z;
        }

        FORCE_INLINE AABB Merge(const AABB& o) const {
            return { VMIN(min,o.min), VMAX(max,o.max) };
        }
    };

    FORCE_INLINE bool RayAABB(const Ray& ray, const AABB& box,
        float& tNear, float& tFar) {
        __m128 ro = _mm_load_ps(&ray.origin._x);
        __m128 rd = _mm_load_ps(&ray.direction._x);
        __m128 bmi = _mm_load_ps(&box.min._x);
        __m128 bma = _mm_load_ps(&box.max._x);

        __m128 inv = _mm_div_ps(_mm_set1_ps(1.f), rd);

        __m128 t0 = _mm_mul_ps(_mm_sub_ps(bmi, ro), inv);
        __m128 t1 = _mm_mul_ps(_mm_sub_ps(bma, ro), inv);

        __m128 tlo = _mm_min_ps(t0, t1);   
        __m128 thi = _mm_max_ps(t0, t1);   

        __m128 tloYZ = _mm_shuffle_ps(tlo, tlo, _MM_SHUFFLE(3, 3, 2, 1));
        __m128 tloX = _mm_shuffle_ps(tlo, tlo, _MM_SHUFFLE(3, 3, 3, 0));
        __m128 tnr = _mm_max_ps(tloX, _mm_max_ps(
            _mm_shuffle_ps(tloYZ, tloYZ, _MM_SHUFFLE(3, 3, 3, 0)),
            _mm_shuffle_ps(tloYZ, tloYZ, _MM_SHUFFLE(3, 3, 3, 1))));

        __m128 thiYZ = _mm_shuffle_ps(thi, thi, _MM_SHUFFLE(3, 3, 2, 1));
        __m128 thiX = _mm_shuffle_ps(thi, thi, _MM_SHUFFLE(3, 3, 3, 0));
        __m128 tfr = _mm_min_ps(thiX, _mm_min_ps(
            _mm_shuffle_ps(thiYZ, thiYZ, _MM_SHUFFLE(3, 3, 3, 0)),
            _mm_shuffle_ps(thiYZ, thiYZ, _MM_SHUFFLE(3, 3, 3, 1))));

        tNear = _mm_cvtss_f32(tnr);
        tFar = _mm_cvtss_f32(tfr);

        return tFar >= tNear && tFar >= 0.f;
    }

    FORCE_INLINE bool RayAABB(const Ray& ray, const AABB& box) {
        float tn, tf;
        return RayAABB(ray, box, tn, tf);
    }

    struct Sphere {
        Vec3  center;
        float radius;
    };

    FORCE_INLINE bool RaySphere(const Ray& ray, const Sphere& sphere,
        float& t) {
        Vec3  oc = SUB(ray.origin, sphere.center);
        float a = DOT(ray.direction, ray.direction);
        float h = DOT(oc, ray.direction);   
        float c = DOT(oc, oc) - sphere.radius * sphere.radius;
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
        float& t, float& u, float& v) {
        Vec3 e1 = SUB(v1, v0);
        Vec3 e2 = SUB(v2, v0);
        Vec3 h = CROSS(ray.direction, e2);
        float a = DOT(e1, h);
        if (std::abs(a) < EPSILON) return false;  

        float f = 1.f / a;
        Vec3 s = SUB(ray.origin, v0);
        u = f * DOT(s, h);
        if (u < 0.f || u > 1.f) return false;

        Vec3 q = CROSS(s, e1);
        v = f * DOT(ray.direction, q);
        if (v < 0.f || u + v > 1.f) return false;

        t = f * DOT(e2, q);
        return t > EPSILON;
    }

    struct Frustum {
        Vec4 planes[6];       
    };

    FORCE_INLINE bool FrustumCullSphere(const Frustum& f,
        const Vec3& center, float radius) {
        __m128 cx = _mm_set1_ps(center._x);
        __m128 cy = _mm_set1_ps(center._y);
        __m128 cz = _mm_set1_ps(center._z);

        for (int i = 0; i < 6; ++i) {
            __m128 p = _mm_load_ps(&f.planes[i]._x);
            __m128 nx = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 ny = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 nz = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 d = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 3, 3, 3));

            float dist = _mm_cvtss_f32(
                _mm_add_ps(d, _mm_add_ps(_mm_mul_ps(nx, cx),
                    _mm_add_ps(_mm_mul_ps(ny, cy),
                        _mm_mul_ps(nz, cz)))));
            if (dist < -radius) return true;  
        }
        return false;
    }

    FORCE_INLINE Vec3 operator+(const Vec3& a, const Vec3& b) { return ADD(a, b); }
    FORCE_INLINE Vec3 operator-(const Vec3& a, const Vec3& b) { return SUB(a, b); }
    FORCE_INLINE Vec3 operator*(const Vec3& a, float s) { return MUL(a, s); }
    FORCE_INLINE Vec3 operator*(float s, const Vec3& a) { return MUL(a, s); }
    FORCE_INLINE Vec3 operator/(const Vec3& a, float s) { return DIV(a, s); }
    FORCE_INLINE Vec3 operator-(const Vec3& a) { return NEG(a); }
    FORCE_INLINE Vec3& operator+=(Vec3& a, const Vec3& b) { a = ADD(a, b); return a; }
    FORCE_INLINE Vec3& operator-=(Vec3& a, const Vec3& b) { a = SUB(a, b); return a; }
    FORCE_INLINE Vec3& operator*=(Vec3& a, float s) { a = MUL(a, s); return a; }

    FORCE_INLINE Vec4 operator+(const Vec4& a, const Vec4& b) { return ADD(a, b); }
    FORCE_INLINE Vec4 operator-(const Vec4& a, const Vec4& b) { return SUB(a, b); }
    FORCE_INLINE Vec4 operator*(const Vec4& a, float s) { return MUL(a, s); }

    FORCE_INLINE Mat4 operator*(const Mat4& a, const Mat4& b) { return MUL(a, b); }
    FORCE_INLINE Vec4 operator*(const Mat4& m, const Vec4& v) { return TransformVec4(m, v); }

    FORCE_INLINE Quat operator*(const Quat& a, const Quat& b) { return QuatMul(a, b); }

    inline bool CPU_HasSSE4() {
#if defined(_MSC_VER)
        int info[4]; __cpuid(info, 1);
        return (info[2] & (1 << 19)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("sse4.1");
#else
        return false;
#endif
    }

    inline bool CPU_HasAVX2() {
#if defined(_MSC_VER)
        int info[4]; __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx2");
#else
        return false;
#endif
    }

}   
