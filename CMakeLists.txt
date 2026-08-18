cmake_minimum_required(VERSION 3.21)
project(amd-hip-p2p-test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path: environment variable or -DROCM_PATH
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    message(FATAL_ERROR "ROCM_PATH not set. Pass -DROCM_PATH=<path> or set env HIP_PATH/ROCM_PATH.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# GPU targets
if(NOT DEFINED AMDGPU_TARGETS)
    set(AMDGPU_TARGETS "gfx1101;gfx1201" CACHE STRING "AMD GPU targets")
endif()

enable_language(HIP)

add_executable(amd-hip-p2p-test src/main.cpp)
target_link_libraries(amd-hip-p2p-test PRIVATE hip::device)
