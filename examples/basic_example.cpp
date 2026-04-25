/*
basic_example.cpp by FLANGZELER

DATE: 04-25-2026 | RELEASE: 1.0.2

LICENSE: This code is released under the MIT License. See LICENSE file for details.

flame (FLANGZELER's MATH ENGINE) is a SIMD based math library made to provide high-performance mathematical operations whether on CPU or GPU.
It can be used for various mathematical computations, including vector and matrix operations, quaternion manipulations, and more.

This file is given to demostrate the basic usage of the flame library, showcasing its basic use.

*/
#include <iostream>
#include "math.h"

using namespace flame;

void PrintVec3(const char* name, const Vec3& v) {
    std::cout << name << ": ("
        << v._x << ", "
        << v._y << ", "
        << v._z << ")\n";
}

void PrintVec4(const char* name, const Vec4& v) {
    std::cout << name << ": ("
        << v._x << ", "
        << v._y << ", "
        << v._z << ", "
        << v._w << ")\n";
}

void PrintMat4(const char* name, const Mat4& m) {
    std::cout << name << ":\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "[ ";
        for (int j = 0; j < 4; j++) {
            std::cout << m.Get(i, j) << " ";
        }
        std::cout << "]\n";
    }
}

int main() {

    std::cout << "===== flame::math verification demo =====\n\n";

    // ============================================================
    // VECTOR OPS
    // ============================================================

    Vec3 a(1, 2, 3);
    Vec3 b(4, 5, 6);

    PrintVec3("A", a);
    PrintVec3("B", b);

    std::cout << "Dot(A,B): " << DOT(a, b) << "\n";

    Vec3 cross = CROSS(a, b);
    PrintVec3("Cross(A,B)", cross);

    Vec3 norm = NORMALIZE(a);
    PrintVec3("Normalize(A)", norm);

    std::cout << "Length(A): " << LENGTH(a) << "\n\n";

    // ============================================================
    // MATRIX + TRANSFORM
    // ============================================================

    Vec3 position(10, 0, 0);
    Quat rotation = QuatFromAxisAngle(Vec3(0, 1, 0), ToRadians(90));
    Vec3 scale(1, 1, 1);

    Mat4 model = TRS(position, rotation, scale);
    PrintMat4("Model Matrix", model);

    Vec3 point(1, 0, 0);
    Vec3 transformed = TransformPoint(model, point);

    PrintVec3("Original Point", point);
    PrintVec3("Transformed Point", transformed);
    std::cout << "\n";

    // ============================================================
    // CAMERA
    // ============================================================

    Mat4 view = LookAt(
        Vec3(0, 0, 5),
        Vec3(0, 0, 0),
        Vec3(0, 1, 0)
    );

    Mat4 proj = Perspective(60.0f, 1.6f, 0.1f, 100.0f);

    Mat4 vp = proj * view;

    PrintMat4("View Matrix", view);
    PrintMat4("Projection Matrix", proj);
    PrintMat4("VP Matrix", vp);
    std::cout << "\n";

    // ============================================================
    // QUATERNION ROTATION
    // ============================================================

    Vec3 dir(1, 0, 0);

    Quat q = QuatFromAxisAngle(Vec3(0, 1, 0), ToRadians(180));
    Vec3 rotated = QuatRotate(q, dir);

    PrintVec3("Original Dir", dir);
    PrintVec3("Rotated Dir (180 Y)", rotated);
    std::cout << "\n";

    // ============================================================
    // RAY vs AABB
    // ============================================================

    Ray ray{ Vec3(0,0,0), NORMALIZE(Vec3(1,0,0)) };
    AABB box = AABB::FromCenterExtents(Vec3(5, 0, 0), Vec3(1, 1, 1));

    float tNear, tFar;
    bool hit = RayAABB(ray, box, tNear, tFar);

    std::cout << "Ray-AABB Hit: " << (hit ? "YES" : "NO") << "\n";
    if (hit) {
        std::cout << "tNear: " << tNear << "\n";
        std::cout << "tFar : " << tFar << "\n";
    }

    std::cout << "\n";

    // ============================================================
    // SoA SIMD TEST (AVX batch)
    // ============================================================

    Vec4SoA soaA, soaB;

    for (int i = 0; i < 8; i++) {
        soaA.Set(i, Vec4(i, i, i, 1));
        soaB.Set(i, Vec4(1, 2, 3, 1));
    }

    Vec4SoA result = AddSoA(soaA, soaB);

    std::cout << "SoA Add result (first 3 lanes):\n";
    for (int i = 0; i < 3; i++) {
        Vec4 v = result.Get(i);
        std::cout << "Lane " << i << ": ("
            << v._x << ", "
            << v._y << ", "
            << v._z << ", "
            << v._w << ")\n";
    }

    std::cout << "\n===== END =====\n";

    return 0;
}
