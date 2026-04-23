#pragma once
#include <immintrin.h>
#include <concepts>
#include <cmath>

namespace flame {

    const float PI = 3.14159265359f;
    inline float ToRadians(float degrees) { return degrees * (PI / 180.0f); }

    struct alignas(16) Vec2 {
        float _x, _y, _z, _w;
        Vec2(float x = 0, float y = 0) : _x(x), _y(y), _z(0.0f), _w(0.0f) {}
    };

    struct alignas(16) Vec3 {
        float _x, _y, _z, _w;
        Vec3(float x = 0, float y = 0, float z = 0) : _x(x), _y(y), _z(z), _w(0.0f) {}
    };

    struct alignas(16) Vec4 {
        float _x, _y, _z, _w;
        Vec4(float x = 0, float y = 0, float z = 0, float w = 1.0f) : _x(x), _y(y), _z(z), _w(w) {}
    };

    struct alignas(16) Quat {
        float _x, _y, _z, _w;
        Quat(float x = 0, float y = 0, float z = 0, float w = 1) : _x(x), _y(y), _z(z), _w(w) {}
    };

    struct alignas(16) Mat4 {
        __m128 rows[4];

        static Mat4 Identity() {
            Mat4 m;
            m.rows[0] = _mm_setr_ps(1, 0, 0, 0);
            m.rows[1] = _mm_setr_ps(0, 1, 0, 0);
            m.rows[2] = _mm_setr_ps(0, 0, 1, 0);
            m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
            return m;
        }
    };

    template <typename T>
    concept vec_type = requires(T v) { { v._x } -> std::same_as<float&>; };

    template <vec_type T>
    inline T ADD(const T& a, const T& b) {
        __m128 res = _mm_add_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x));
        T out; _mm_store_ps(&out._x, res); return out;
    }

    template <vec_type T>
    inline T SUB(const T& a, const T& b) {
        __m128 res = _mm_sub_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x));
        T out; _mm_store_ps(&out._x, res); return out;
    }

    template <vec_type T>
    inline T MUL(const T& a, float s) {
        __m128 res = _mm_mul_ps(_mm_load_ps(&a._x), _mm_set1_ps(s));
        T out; _mm_store_ps(&out._x, res); return out;
    }

    template <vec_type T>
    inline float DOT(const T& a, const T& b) {
        __m128 res = _mm_dp_ps(_mm_load_ps(&a._x), _mm_load_ps(&b._x), 0x71);
        return _mm_cvtss_f32(res);
    }

    template <vec_type T>
    inline T NORMALIZE(const T& a) {
        __m128 reg = _mm_load_ps(&a._x);
        __m128 lsq = _mm_dp_ps(reg, reg, 0x7F);
        __m128 invLen = _mm_rsqrt_ps(lsq);
        T out; _mm_store_ps(&out._x, _mm_mul_ps(reg, invLen));
        return out;
    }

    inline Vec3 CROSS(const Vec3& a, const Vec3& b) {
        __m128 rA = _mm_load_ps(&a._x); __m128 rB = _mm_load_ps(&b._x);
        __m128 t0 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 t1 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 t2 = _mm_shuffle_ps(rA, rA, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 t3 = _mm_shuffle_ps(rB, rB, _MM_SHUFFLE(3, 0, 2, 1));
        Vec3 out; _mm_store_ps(&out._x, _mm_sub_ps(_mm_mul_ps(t0, t1), _mm_mul_ps(t2, t3)));
        return out;
    }

    inline Mat4 MUL(const Mat4& a, const Mat4& b) {
        Mat4 out;
        for (int i = 0; i < 4; i++) {
            __m128 row = a.rows[i];
            __m128 res = _mm_mul_ps(_mm_set1_ps(((float*)&row)[0]), b.rows[0]);
            res = _mm_add_ps(res, _mm_mul_ps(_mm_set1_ps(((float*)&row)[1]), b.rows[1]));
            res = _mm_add_ps(res, _mm_mul_ps(_mm_set1_ps(((float*)&row)[2]), b.rows[2]));
            res = _mm_add_ps(res, _mm_mul_ps(_mm_set1_ps(((float*)&row)[3]), b.rows[3]));
            out.rows[i] = res;
        }
        return out;
    }

    inline Mat4 Translate(Vec3 v) {
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(1, 0, 0, v._x);
        m.rows[1] = _mm_setr_ps(0, 1, 0, v._y);
        m.rows[2] = _mm_setr_ps(0, 0, 1, v._z);
        return m;
    }

    inline Mat4 Scale(Vec3 v) {
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(v._x, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, v._y, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, v._z, 0);
        return m;
    }

    inline Mat4 RotateY(float angle) {
        float s = std::sin(angle); float c = std::cos(angle);
        Mat4 m = Mat4::Identity();
        m.rows[0] = _mm_setr_ps(c, 0, s, 0);
        m.rows[2] = _mm_setr_ps(-s, 0, c, 0);
        return m;
    }

    inline Mat4 Perspective(float fovDeg, float aspect, float nearP, float farP) {
        float f = 1.0f / std::tan(ToRadians(fovDeg) * 0.5f);
        Mat4 m;
        m.rows[0] = _mm_setr_ps(f / aspect, 0, 0, 0);
        m.rows[1] = _mm_setr_ps(0, f, 0, 0);
        m.rows[2] = _mm_setr_ps(0, 0, farP / (farP - nearP), -(farP * nearP) / (farP - nearP));
        m.rows[3] = _mm_setr_ps(0, 0, 1, 0);
        return m;
    }

    inline Mat4 LookAt(Vec3 eye, Vec3 target, Vec3 up) {
        Vec3 z = NORMALIZE(SUB(target, eye));
        Vec3 x = NORMALIZE(CROSS(up, z));
        Vec3 y = CROSS(z, x);
        Mat4 m;
        m.rows[0] = _mm_setr_ps(x._x, x._y, x._z, -DOT(x, eye));
        m.rows[1] = _mm_setr_ps(y._x, y._y, y._z, -DOT(y, eye));
        m.rows[2] = _mm_setr_ps(z._x, z._y, z._z, -DOT(z, eye));
        m.rows[3] = _mm_setr_ps(0, 0, 0, 1);
        return m;
    }
    inline Mat4 Transpose(const Mat4& m) {
        Mat4 out = m;
        _MM_TRANSPOSE4_PS(out.rows[0], out.rows[1], out.rows[2], out.rows[3]);
        return out;
    }

    inline Mat4 Inverse(const Mat4& m) {
        __m128 ra = m.rows[0];
        __m128 rb = m.rows[1];
        __m128 rc = m.rows[2];
        __m128 rd = m.rows[3];

        _MM_TRANSPOSE4_PS(ra, rb, rc, rd);

        __m128 tmp1 = _mm_mul_ps(rc, rd);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
        __m128 row0 = _mm_mul_ps(rb, tmp1);
        __m128 row1 = _mm_mul_ps(ra, tmp1);
        tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
        row0 = _mm_sub_ps(_mm_mul_ps(rb, tmp1), row0);
        row1 = _mm_sub_ps(row1, _mm_mul_ps(ra, tmp1));
        row1 = _mm_shuffle_ps(row1, row1, 0x4E);

        float det = ((float*)&ra)[0] * ((float*)&row0)[0] + ((float*)&ra)[1] * ((float*)&row0)[1] +
            ((float*)&ra)[2] * ((float*)&row0)[2] + ((float*)&ra)[3] * ((float*)&row0)[3];

        float invDet = 1.0f / det;
        Mat4 res;
        for (int i = 0; i < 4; i++) res.rows[i] = _mm_mul_ps(row0, _mm_set1_ps(invDet));    
        return res;
    }

    inline float Determinant(const Mat4& m) {
        __m128 res = _mm_dp_ps(m.rows[0], m.rows[0], 0xFF);
        return _mm_cvtss_f32(res);
    }
}   