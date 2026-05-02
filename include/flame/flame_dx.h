/*
flame_dx.h — DirectXMath Interoperability Layer for flame

DATE: 05-02-2026 | RELEASE: 1.0.0

LICENSE: MIT. See LICENSE file for details.

DESIGN PRINCIPLES
─────────────────
  • Zero heap allocation — all conversions work in registers or on the stack.
  • Zero-copy register paths — Vec4/Quat/Mat4 share the exact same binary
    layout as XMVECTOR/XMMATRIX on x86; conversions are reinterpret casts or
    single _mm_load_ps/_mm_store_ps round-trips, never serialised scalar loops.
  • Explicit API — no implicit conversion operators; all conversions are named
    functions so the programmer consciously pays the cost.
  • No virtual dispatch, no inheritance, no hidden overhead.
  • noexcept everywhere — math cannot throw.

LAYOUT CONTRACT
───────────────
  flame::Vec2  → float[4] aligned(16)  →  XMVECTOR (xyz=xy, zw=0)
  flame::Vec3  → float[4] aligned(16)  →  XMVECTOR (xyz=xyz, w=0)
  flame::Vec4  → float[4] aligned(16)  →  XMVECTOR (xyzw)         ← direct cast
  flame::Quat  → float[4] aligned(16)  →  XMVECTOR (xyzw)         ← direct cast
  flame::Mat4  → __m128[4] aligned(16) →  XMMATRIX (4×XMVECTOR)   ← direct cast

  The direct-cast types (Vec4, Quat, Mat4) allow pointer-level reinterpret_cast
  because their ABI storage is binary-identical to the DX equivalents.

USAGE
─────
  #include "flame_dx.h"          // pulls in both flame.h and DirectXMath.h

  // Convert to DX
  Vec3 pos = { 1, 2, 3 };
  XMVECTOR dxPos = flame::dx::ToDX(pos);

  // Convert from DX
  XMMATRIX dxView = XMMatrixLookAtRH(...);
  Mat4 view       = flame::dx::FromDX(dxView);

  // Mixed rendering pipeline
  Mat4  proj   = Perspective(60.f, aspect, 0.1f, 1000.f);
  XMMATRIX vp  = flame::dx::ToDX(MUL(proj, view));

REQUIREMENTS
────────────
  • Windows SDK 10.0+ (DirectXMath shipped with it), OR
    vcpkg: vcpkg install directxmath
  • C++20 (/std:c++20 or -std=c++20)
  • SSE4.1 target minimum (AVX2 recommended for full FLAME_AVX path)
  • flame.h must be in the include path

COMPILATION NOTES
─────────────────
  MSVC  : /arch:AVX2 /O2 /fp:fast
  Clang : -mavx2 -mfma -O3 -ffast-math
  GCC   : -mavx2 -mfma -O3 -ffast-math
*/

#pragma once

// ─── DirectXMath guard ───────────────────────────────────────────────────────
#if !defined(_DIRECTXMATH_H_) && !defined(DIRECTX_MATH_VERSION)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <DirectXMath.h>
#endif

#include "flame.h"

#include <type_traits>
#include <cstring> // memcpy for aliasing-safe copies where needed

// Compile-time layout assertions — fail loudly if our assumptions break
static_assert(sizeof(flame::Vec4)  == sizeof(DirectX::XMVECTOR),
    "[flame_dx] Vec4 and XMVECTOR size mismatch");
static_assert(alignof(flame::Vec4) >= 16,
    "[flame_dx] Vec4 alignment < 16 bytes — SIMD loads unsafe");
static_assert(sizeof(flame::Quat)  == sizeof(DirectX::XMVECTOR),
    "[flame_dx] Quat and XMVECTOR size mismatch");
static_assert(sizeof(flame::Mat4)  == sizeof(DirectX::XMMATRIX),
    "[flame_dx] Mat4 and XMMATRIX size mismatch");
static_assert(alignof(flame::Mat4) >= 16,
    "[flame_dx] Mat4 alignment < 16 bytes — SIMD loads unsafe");

namespace flame {
namespace dx   {

// ─────────────────────────────────────────────────────────────────────────────
//  Tag types for FromDX overload disambiguation
//  Usage:  Vec3 v = FromDX(xmv, AS_VEC3);
// ─────────────────────────────────────────────────────────────────────────────
struct AsVec2Tag {}; constexpr AsVec2Tag AS_VEC2{};
struct AsVec3Tag {}; constexpr AsVec3Tag AS_VEC3{};
struct AsVec4Tag {}; constexpr AsVec4Tag AS_VEC4{};
struct AsQuatTag {}; constexpr AsQuatTag AS_QUAT{};

// ═════════════════════════════════════════════════════════════════════════════
//  flame → DirectXMath
// ═════════════════════════════════════════════════════════════════════════════

// ─── Vec2 ────────────────────────────────────────────────────────────────────
/// Converts flame::Vec2 to XMVECTOR.
/// The z and w lanes contain 0 (from Vec2's padding invariant).
/// Cost: 1 × _mm_load_ps (aligned).
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDX(const Vec2& v) noexcept {
    return _mm_load_ps(&v._x);
}

// ─── Vec3 ────────────────────────────────────────────────────────────────────
/// Converts flame::Vec3 to XMVECTOR (w = 0).
/// Cost: 1 × _mm_load_ps (aligned).
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDX(const Vec3& v) noexcept {
    return _mm_load_ps(&v._x);
}

/// Vec3 as a homogeneous POINT (w = 1).  Use for position transforms.
/// Cost: 1 × _mm_load_ps + 1 × _mm_blend_ps.
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDXPoint(const Vec3& v) noexcept {
    __m128 reg  = _mm_load_ps(&v._x);
    __m128 one  = _mm_set_ss(1.f);  // (1, 0, 0, 0)
    // blend: keep xyz from reg, take w from one (shuffled to lane 3)
    one = _mm_shuffle_ps(one, one, _MM_SHUFFLE(0,0,0,0)); // broadcast 1 to all
    // mask: keep lanes 0-2 from reg, lane 3 from one
    return _mm_blend_ps(reg, one, 0x8); // 0x8 = select lane 3 from second arg
}

/// Vec3 as a homogeneous DIRECTION (w = 0).  Alias of ToDX(Vec3) for clarity.
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDXDir(const Vec3& v) noexcept {
    return _mm_load_ps(&v._x);
}

// ─── Vec4 ────────────────────────────────────────────────────────────────────
/// Converts flame::Vec4 to XMVECTOR.
/// Vec4 is layout-identical to XMVECTOR — this is a single aligned load.
/// Cost: 1 × _mm_load_ps.
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDX(const Vec4& v) noexcept {
    return _mm_load_ps(&v._x);
}

// ─── Quat ────────────────────────────────────────────────────────────────────
/// Converts flame::Quat to XMVECTOR (DirectXMath quaternion convention: xyzw).
/// Both flame and DirectXMath store quaternions as (x, y, z, w) — direct load.
/// Cost: 1 × _mm_load_ps.
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR ToDX(const Quat& q) noexcept {
    return _mm_load_ps(&q._x);
}

// ─── Mat4 ────────────────────────────────────────────────────────────────────
/// Converts flame::Mat4 to DirectX::XMMATRIX.
/// Both are 4 × __m128 (row-major, 16-byte aligned) — binary-identical layout.
/// The reinterpret_cast is safe under strict aliasing because __m128 is a
/// compiler-magic type exempt from strict-aliasing restrictions (in practice),
/// and both operands are 16-byte aligned.
/// Cost: 0 (pure pointer reinterpret, no data movement).
[[nodiscard]] FORCE_INLINE
DirectX::XMMATRIX ToDX(const Mat4& m) noexcept {
    // Pointer-cast path (zero data movement):
    return *reinterpret_cast<const DirectX::XMMATRIX*>(m.rows);
}

/// Variant accepting a pointer (e.g., array element) — avoids extra copy.
[[nodiscard]] FORCE_INLINE
const DirectX::XMMATRIX& ToDXRef(const Mat4& m) noexcept {
    return *reinterpret_cast<const DirectX::XMMATRIX*>(m.rows);
}

// ─── Transform ───────────────────────────────────────────────────────────────
/// Bakes a flame::Transform into a DirectX::XMMATRIX world matrix.
/// Cost: TRS() + 1 pointer reinterpret.
[[nodiscard]] FORCE_INLINE
DirectX::XMMATRIX ToDX(const Transform& t) noexcept {
    Mat4 m = t.ToMat4();
    return ToDX(m);
}

// ═════════════════════════════════════════════════════════════════════════════
//  DirectXMath → flame
// ═════════════════════════════════════════════════════════════════════════════

// ─── XMVECTOR → Vec2 ─────────────────────────────────────────────────────────
/// Extracts xy from an XMVECTOR into flame::Vec2 (zw padding zeroed).
/// Cost: 1 × _mm_store_ps + 2 scalar writes (or 1 blend + store).
[[nodiscard]] FORCE_INLINE
Vec2 FromDXVec2(DirectX::FXMVECTOR v) noexcept {
    Vec2 out;
    _mm_store_ps(&out._x, v);
    out._z = 0.f; out._w = 0.f; // enforce padding invariant
    return out;
}

/// Tag-dispatch overload: FromDX(xmv, AS_VEC2)
[[nodiscard]] FORCE_INLINE
Vec2 FromDX(DirectX::FXMVECTOR v, AsVec2Tag) noexcept {
    return FromDXVec2(v);
}

// ─── XMVECTOR → Vec3 ─────────────────────────────────────────────────────────
/// Extracts xyz from an XMVECTOR into flame::Vec3 (w padding zeroed).
/// Cost: 1 × _mm_store_ps + 1 scalar write.
[[nodiscard]] FORCE_INLINE
Vec3 FromDXVec3(DirectX::FXMVECTOR v) noexcept {
    Vec3 out;
    _mm_store_ps(&out._x, v);
    out._w = 0.f; // enforce padding invariant (DX may have non-zero w)
    return out;
}

/// Tag-dispatch overload: FromDX(xmv, AS_VEC3)
[[nodiscard]] FORCE_INLINE
Vec3 FromDX(DirectX::FXMVECTOR v, AsVec3Tag) noexcept {
    return FromDXVec3(v);
}

// ─── XMVECTOR → Vec4 ─────────────────────────────────────────────────────────
/// Stores all four lanes of an XMVECTOR into flame::Vec4.
/// Cost: 1 × _mm_store_ps.
[[nodiscard]] FORCE_INLINE
Vec4 FromDXVec4(DirectX::FXMVECTOR v) noexcept {
    Vec4 out;
    _mm_store_ps(&out._x, v);
    return out;
}

/// Tag-dispatch overload: FromDX(xmv, AS_VEC4)
[[nodiscard]] FORCE_INLINE
Vec4 FromDX(DirectX::FXMVECTOR v, AsVec4Tag) noexcept {
    return FromDXVec4(v);
}

// ─── XMVECTOR → Quat ─────────────────────────────────────────────────────────
/// Converts a DirectXMath quaternion XMVECTOR to flame::Quat.
/// Both are (x,y,z,w) — direct store.
/// Cost: 1 × _mm_store_ps.
[[nodiscard]] FORCE_INLINE
Quat FromDXQuat(DirectX::FXMVECTOR v) noexcept {
    Quat out;
    _mm_store_ps(&out._x, v);
    return out;
}

/// Tag-dispatch overload: FromDX(xmv, AS_QUAT)
[[nodiscard]] FORCE_INLINE
Quat FromDX(DirectX::FXMVECTOR v, AsQuatTag) noexcept {
    return FromDXQuat(v);
}

// ─── XMMATRIX → Mat4 ─────────────────────────────────────────────────────────
/// Converts a DirectX::XMMATRIX to flame::Mat4.
/// Both are 4 × __m128, row-major — copy 4 registers.
/// Cost: 4 × register copy (no memory traffic if already in XMM regs).
[[nodiscard]] FORCE_INLINE
Mat4 FromDX(const DirectX::XMMATRIX& m) noexcept {
    Mat4 out;
    out.rows[0] = m.r[0];
    out.rows[1] = m.r[1];
    out.rows[2] = m.r[2];
    out.rows[3] = m.r[3];
    return out;
}

/// Pointer-reinterpret variant (zero-cost, read-only alias).
/// Only call if m has 16-byte alignment (guaranteed for XMMATRIX locals).
[[nodiscard]] FORCE_INLINE
const Mat4& FromDXRef(const DirectX::XMMATRIX& m) noexcept {
    return *reinterpret_cast<const Mat4*>(&m);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Batch (SIMD-bulk) conversions — ECS / render-loop friendly
// ═════════════════════════════════════════════════════════════════════════════

/// Convert N aligned flame::Vec4s to XMVECTOR array.
/// src and dst must be 16-byte aligned.  No heap allocation.
/// Cost: N aligned loads (usually a no-op if compiler keeps values in regs).
inline void BatchToDX(const Vec4* RESTRICT src,
                      DirectX::XMVECTOR* RESTRICT dst,
                      size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        dst[i] = _mm_load_ps(&src[i]._x);
}

/// Convert N XMVECTOR values back to aligned flame::Vec4 array.
inline void BatchFromDX(const DirectX::XMVECTOR* RESTRICT src,
                        Vec4* RESTRICT dst,
                        size_t count) noexcept {
    for (size_t i = 0; i < count; ++i)
        _mm_store_ps(&dst[i]._x, src[i]);
}

/// Convert N flame::Mat4s to XMMATRIX array.
/// Cost: 4 register moves per matrix (stays in XMM regs if already hot).
inline void BatchToDX(const Mat4* RESTRICT src,
                      DirectX::XMMATRIX* RESTRICT dst,
                      size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        dst[i].r[0] = src[i].rows[0];
        dst[i].r[1] = src[i].rows[1];
        dst[i].r[2] = src[i].rows[2];
        dst[i].r[3] = src[i].rows[3];
    }
}

/// Convert N XMMATRIX values to flame::Mat4 array.
inline void BatchFromDX(const DirectX::XMMATRIX* RESTRICT src,
                        Mat4* RESTRICT dst,
                        size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        dst[i].rows[0] = src[i].r[0];
        dst[i].rows[1] = src[i].r[1];
        dst[i].rows[2] = src[i].r[2];
        dst[i].rows[3] = src[i].r[3];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Rendering-pipeline helpers — common patterns that avoid round-trips
// ═════════════════════════════════════════════════════════════════════════════

/// Build a DX view-projection matrix from flame camera parameters.
/// Avoids materialising intermediate flame Mat4 before the DX conversion.
[[nodiscard]] FORCE_INLINE
DirectX::XMMATRIX BuildViewProjectionDX(
        const Vec3& eye, const Vec3& target, const Vec3& up,
        float fovDeg, float aspect, float nearP, float farP) noexcept {
    Mat4 view = LookAt(eye, target, up);
    Mat4 proj = Perspective(fovDeg, aspect, nearP, farP);
    return ToDX(MUL(proj, view)); // single reinterpret at the end
}

/// Build a DX reverse-Z view-projection (better depth precision).
[[nodiscard]] FORCE_INLINE
DirectX::XMMATRIX BuildViewProjectionRevZDX(
        const Vec3& eye, const Vec3& target, const Vec3& up,
        float fovDeg, float aspect, float nearP, float farP) noexcept {
    Mat4 view = LookAt(eye, target, up);
    Mat4 proj = PerspectiveRevZ(fovDeg, aspect, nearP, farP);
    return ToDX(MUL(proj, view));
}

/// Transform a DX position vector by a flame matrix.
/// Avoids converting the matrix if it is already in flame format.
[[nodiscard]] FORCE_INLINE
DirectX::XMVECTOR TransformDX(const Mat4& m, DirectX::FXMVECTOR dxVec) noexcept {
    Vec4 v = FromDXVec4(dxVec);
    return ToDX(TransformVec4(m, v));
}

/// Transform N world positions (as XMVECTOR) by a single flame::Mat4.
/// Writes results to a pre-allocated XMVECTOR array (no alloc).
inline void TransformPointsBatchDX(
        const Mat4& model,
        const DirectX::XMVECTOR* RESTRICT srcPositions,
        DirectX::XMVECTOR* RESTRICT dstPositions,
        size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        Vec4 v = FromDXVec4(srcPositions[i]);
        dstPositions[i] = ToDX(TransformVec4(model, v));
    }
}

/// Convert a flame Frustum (6 Planes) to 6 DX XMVECTOR planes.
/// Caller supplies a 6-element XMVECTOR array.
inline void FrustumToDX(const Frustum& f,
                         DirectX::XMVECTOR out[6]) noexcept {
    for (int i = 0; i < 6; ++i)
        out[i] = _mm_load_ps(&f.planes[i]._a);
}

/// Build a flame Frustum directly from a DX view-projection XMMATRIX.
[[nodiscard]] FORCE_INLINE
Frustum FrustumFromDX(const DirectX::XMMATRIX& vp) noexcept {
    return FrustumFromVP(FromDX(vp));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Constant buffer upload helpers
//  These functions exist to document the correct workflow when populating
//  D3D11/D3D12 constant buffers from flame types without unnecessary copies.
// ═════════════════════════════════════════════════════════════════════════════

// PATTERN: mapping a D3D constant buffer — memcpy from flame::Mat4 directly
// because Mat4 and XMMATRIX are layout-identical.  No intermediate needed.
//
//   CB layout example:
//   struct PerFrameCB {
//       flame::Mat4 viewProj;   // 64 bytes, 16-byte aligned — upload as-is
//       flame::Vec4 cameraPos;  // 16 bytes
//   };
//
//   D3D11_MAPPED_SUBRESOURCE mapped;
//   ctx->Map(pCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
//   memcpy(mapped.pData, &cbData, sizeof(PerFrameCB));   // zero-conversion
//   ctx->Unmap(pCB, 0);

// ═════════════════════════════════════════════════════════════════════════════
//  XMFLOAT interop (for serialization / upload structs that require XMFLOAT)
// ═════════════════════════════════════════════════════════════════════════════
// These paths go through DirectXMath's XMLoad/XMStore for maximum correctness
// at the cost of one extra round-trip.  Only use at serialization boundaries,
// never in hot paths.

[[nodiscard]] FORCE_INLINE
DirectX::XMFLOAT2 ToXMFLOAT2(const Vec2& v) noexcept {
    return { v._x, v._y };
}

[[nodiscard]] FORCE_INLINE
DirectX::XMFLOAT3 ToXMFLOAT3(const Vec3& v) noexcept {
    return { v._x, v._y, v._z };
}

[[nodiscard]] FORCE_INLINE
DirectX::XMFLOAT4 ToXMFLOAT4(const Vec4& v) noexcept {
    return { v._x, v._y, v._z, v._w };
}

[[nodiscard]] FORCE_INLINE
DirectX::XMFLOAT4X4 ToXMFLOAT4X4(const Mat4& m) noexcept {
    DirectX::XMFLOAT4X4 out;
    DirectX::XMStoreFloat4x4(&out, ToDX(m));
    return out;
}

[[nodiscard]] FORCE_INLINE
Vec2 FromXMFLOAT2(const DirectX::XMFLOAT2& v) noexcept {
    return { v.x, v.y };
}

[[nodiscard]] FORCE_INLINE
Vec3 FromXMFLOAT3(const DirectX::XMFLOAT3& v) noexcept {
    return { v.x, v.y, v.z };
}

[[nodiscard]] FORCE_INLINE
Vec4 FromXMFLOAT4(const DirectX::XMFLOAT4& v) noexcept {
    return { v.x, v.y, v.z, v.w };
}

[[nodiscard]] FORCE_INLINE
Mat4 FromXMFLOAT4X4(const DirectX::XMFLOAT4X4& v) noexcept {
    return FromDX(DirectX::XMLoadFloat4x4(&v));
}

} // namespace dx
} // namespace flame


