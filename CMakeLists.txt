cmake_minimum_required(VERSION 3.10)

project(Flame)

# Header-only library
add_library(Flame INTERFACE)

# Include directory
target_include_directories(Flame INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
