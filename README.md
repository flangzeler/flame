# Flame🔥

![C++](https://img.shields.io/badge/language-C++-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Header Only](https://img.shields.io/badge/type-header--only-orange)
**Header-only SIMD math library for real-time engines**

Designed for:
- Game Engines
- Rendering Pipelines
- Physics Systems
- High-performance tools

---

## 🚀 Features

### ✅ SIMD-first architecture
- SSE4.1 + AVX2 support
- Automatic fallback
- Zero-overhead abstraction

### ✅ Vector Math
- Vec2 / Vec3 / Vec4
- Dot, Cross, Normalize
- Lerp, Clamp, Min/Max

### ✅ Quaternion System
- Rotation (QuatMul)
- Slerp (high precision)
- Axis-angle conversion
- Vector rotation

### ✅ Matrix System (Mat4)
- Transform (TRS)
- Inverse (fully correct SIMD implementation)
- Determinant
- Transpose
- TransformPoint / TransformDir

### ✅ Camera & Projection
- Perspective
- Orthographic
- LookAt

### ✅ Geometry & Collision
- Ray vs AABB (SIMD optimized)
- Ray vs Sphere
- Ray vs Triangle (Möller–Trumbore)
- Frustum Culling

### ✅ SoA SIMD Batch Processing
- Vec4SoA (8-wide AVX)
- Batch Add / Mul / Dot

---

## 📦 Installation



```cmake
add_subdirectory(flame_math)
target_link_libraries(your_project PRIVATE flame_math)
