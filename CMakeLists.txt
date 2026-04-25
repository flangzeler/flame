cmake_minimum_required(VERSION 3.16)
project(flame_math VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# Options
# ============================================================

option(FLAME_ENABLE_AVX "Enable AVX/AVX2 optimizations" ON)
option(FLAME_BUILD_EXAMPLES "Build examples" ON)

# ============================================================
# Library (Header-only)
# ============================================================

add_library(flame_math INTERFACE)

target_include_directories(flame_math
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(flame_math INTERFACE cxx_std_20)

# SIMD flags
if(FLAME_ENABLE_AVX)
    if(MSVC)
        target_compile_options(flame_math INTERFACE /arch:AVX2)
    else()
        target_compile_options(flame_math INTERFACE -mavx2 -mfma)
    endif()
endif()

# ============================================================
# Example
# ============================================================

if(FLAME_BUILD_EXAMPLES)
    add_executable(example example.cpp)
    target_link_libraries(example PRIVATE flame_math)
endif()

# ============================================================
# Install
# ============================================================

install(DIRECTORY ./
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS flame_math EXPORT flameTargets)

install(EXPORT flameTargets
    FILE flameTargets.cmake
    NAMESPACE flame::
    DESTINATION lib/cmake/flame_math
)
