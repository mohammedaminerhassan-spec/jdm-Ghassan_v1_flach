cmake_minimum_required(VERSION 3.20)

project(ghassan_ai LANGUAGES CXX VERSION 0.3.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    set(VcpkgApplocalDeps OFF CACHE BOOL "Disable vcpkg applocal dependencies copy" FORCE)
    set(CMAKE_VS_GLOBALS "VcpkgApplocalDeps=false")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Git QUIET)
set(GAI_GIT_COMMIT "unknown")
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE GAI_GIT_COMMIT_OUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(GAI_GIT_COMMIT_OUT)
        set(GAI_GIT_COMMIT "${GAI_GIT_COMMIT_OUT}")
    endif()
endif()
message(STATUS "[ghassan-ai] the commit: ${GAI_GIT_COMMIT}")
add_compile_definitions(GAI_GIT_COMMIT="${GAI_GIT_COMMIT}")

# ══════════════════════════════════════════════════════════════════════════════
# Options
# ══════════════════════════════════════════════════════════════════════════════
option(GAI_ENABLE_CUDA    "Build NVIDIA CUDA backend"                       ON)
option(GAI_ENABLE_HIP     "Build AMD ROCm/HIP backend"                      OFF)
option(GAI_ENABLE_SYCL    "Build Intel oneAPI/SYCL backend"                 OFF)
option(GAI_ENABLE_XLA     "Build Google XLA/TPU backend"                    OFF)
option(GAI_ENABLE_OPENMP  "Use OpenMP for CPU kernels"                      ON)
option(GAI_NATIVE_ARCH    "Compile with -march=native (enables AVX-512 etc)" OFF)
option(GAI_FAST_MATH      "CUDA --use_fast_math (faster transcendental)"    OFF)
option(GAI_LTO            "Link-time optimization across translation units"  OFF)
option(GAI_ENABLE_NCCL    "Build NCCL multi-GPU support (CUDA)"             ON)
option(GAI_ENABLE_RCCL    "Build RCCL multi-GPU support (HIP/AMD)"          OFF)
option(GAI_ENABLE_MPI     "Build MPI multi-CPU (DDP) support"               OFF)
option(GAI_ENABLE_PARQUET "Link optional Arrow/Parquet libraries"            OFF)

function(gai_apply_cxx_flags target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W3 /WX /wd5105 /permissive- /Zc:preprocessor /bigobj /utf-8 /MP)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
        if(GAI_LTO)
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/GL>)
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:/LTCG>)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror
            -Wno-unused-parameter -Wno-missing-field-initializers
            -Wunused-but-set-variable -Wunused-variable)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
        if(GAI_NATIVE_ARCH)
            # -march=native: enables AVX2, AVX-512, F16C, VNNI automatically
            target_compile_options(${target} PRIVATE -march=native)
            message(STATUS "[ghassan-ai] native arch: ON (auto-enables AVX-512/VNNI if CPU supports)")
        endif()
        if(GAI_LTO)
            target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-flto>)
            target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-flto>)
        endif()
    endif()
endfunction()

# ══════════════════════════════════════════════════════════════════════════════
# OpenMP
# ══════════════════════════════════════════════════════════════════════════════
if(GAI_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "[ghassan-ai] OpenMP: enabled")
    else()
        message(STATUS "[ghassan-ai] OpenMP: not found (single-threaded CPU kernels)")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA CUDA backend
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_CUDA OFF)
if(GAI_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
            set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real)
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
            endif()
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120)
            elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
            endif()
        endif()
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            message(STATUS "[ghassan-ai] CUDA toolkit not found -> CPU-only build")
        else()
            set(GAI_HAVE_CUDA ON)
            message(STATUS "[ghassan-ai] CUDA: ${CMAKE_CUDA_COMPILER_VERSION} arch=${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(STATUS "[ghassan-ai] CUDA: no toolkit found -> CPU-only build")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# AMD ROCm/HIP backend
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_HIP OFF)
if(GAI_ENABLE_HIP)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)
        find_package(hipblas QUIET)
        find_package(rocblas QUIET)
        if(hipblas_FOUND)
            set(GAI_HAVE_HIP ON)
            message(STATUS "[ghassan-ai] HIP: ${CMAKE_HIP_COMPILER_VERSION}")
            message(STATUS "[ghassan-ai] hipBLAS: found")
        else()
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(HIPBLAS QUIET hipblas)
                if(HIPBLAS_FOUND)
                    set(GAI_HAVE_HIP ON)
                    message(STATUS "[ghassan-ai] HIP: enabled via pkg-config")
                endif()
            endif()
        endif()
        if(NOT GAI_HAVE_HIP)
            message(STATUS "[ghassan-ai] HIP: compiler found but hipBLAS missing -> HIP disabled")
        endif()
    else()
        message(STATUS "[ghassan-ai] HIP: no hipcc found -> AMD GPU disabled")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Intel oneAPI/SYCL backend
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_SYCL OFF)
if(GAI_ENABLE_SYCL)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-fsycl")
        check_cxx_source_compiles(
            "#include <sycl/sycl.hpp>\nint main(){return 0;}"
            GAI_SYCL_COMPILES)
        unset(CMAKE_REQUIRED_FLAGS)
        if(GAI_SYCL_COMPILES)
            find_package(oneMKL QUIET)
            if(oneMKL_FOUND OR MKL_FOUND)
                set(GAI_HAVE_SYCL ON)
                message(STATUS "[ghassan-ai] SYCL/oneAPI: enabled (Intel GPU + oneMKL)")
            else()
                message(STATUS "[ghassan-ai] SYCL: compiler OK but oneMKL not found -> SYCL disabled")
            endif()
        else()
            message(STATUS "[ghassan-ai] SYCL: compiler does not support -fsycl -> disabled")
        endif()
    else()
        message(STATUS "[ghassan-ai] SYCL: requires icpx or clang++ with SYCL support")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Google XLA/TPU backend
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_XLA OFF)
if(GAI_ENABLE_XLA)
    find_path(XLA_INCLUDE_DIR "xla/pjrt/c/pjrt_c_api.h"
        HINTS $ENV{XLA_DIR}/include /usr/local/include /opt/xla/include)
    find_library(XLA_LIBRARY NAMES xla_extension pjrt_c_api_cpu_plugin
        HINTS $ENV{XLA_DIR}/lib /usr/local/lib /opt/xla/lib)
    if(XLA_INCLUDE_DIR AND XLA_LIBRARY)
        set(GAI_HAVE_XLA ON)
        message(STATUS "[ghassan-ai] XLA: found at ${XLA_INCLUDE_DIR}")
        find_library(TPU_LIBRARY NAMES tpu_client
            HINTS $ENV{TPU_SDK_DIR}/lib /usr/local/lib)
        if(TPU_LIBRARY)
            message(STATUS "[ghassan-ai] XLA TPU plugin: ${TPU_LIBRARY}")
        else()
            message(STATUS "[ghassan-ai] XLA: CPU/GPU only (TPU plugin not found)")
        endif()
    else()
        message(STATUS "[ghassan-ai] XLA: not found (set XLA_DIR env var or install openxla)")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Core library sources
# ══════════════════════════════════════════════════════════════════════════════
file(GLOB_RECURSE GAI_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/core/*.cpp
     ${CMAKE_SOURCE_DIR}/tokenizer/*.cpp
     ${CMAKE_SOURCE_DIR}/model/*.cpp
     ${CMAKE_SOURCE_DIR}/format/*.cpp
     ${CMAKE_SOURCE_DIR}/training/*.cpp
     ${CMAKE_SOURCE_DIR}/inference/*.cpp
     ${CMAKE_SOURCE_DIR}/quantization/*.cpp
     ${CMAKE_SOURCE_DIR}/projector/*.cpp
     ${CMAKE_SOURCE_DIR}/evaluation/*.cpp)

# Backends
file(GLOB GAI_BACKEND_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/backends/*.cpp)
list(APPEND GAI_SOURCES ${GAI_BACKEND_SOURCES})

add_library(gai_core STATIC ${GAI_SOURCES})
target_include_directories(gai_core PUBLIC ${CMAKE_SOURCE_DIR})
gai_apply_cxx_flags(gai_core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gai_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gai_core PUBLIC GAI_OPENMP=1)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA CUDA library
# ══════════════════════════════════════════════════════════════════════════════
if(GAI_HAVE_CUDA)
    file(GLOB GAI_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cuda/*.cu)
    add_library(gai_cuda STATIC ${GAI_CUDA_SOURCES})
    target_include_directories(gai_cuda PUBLIC ${CMAKE_SOURCE_DIR})
    set_target_properties(gai_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(gai_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(gai_cuda PUBLIC GAI_CUDA=1)
    target_link_libraries(gai_cuda PUBLIC CUDA::cudart CUDA::cublas)
    target_link_libraries(gai_core PUBLIC gai_cuda)
    target_compile_definitions(gai_core PUBLIC GAI_CUDA=1)
    if(GAI_FAST_MATH)
        target_compile_options(gai_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    if(GAI_LTO)
        set_property(TARGET gai_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    option(GAI_CUDA_WERROR "Fail the CUDA build on device-code warnings" OFF)
    if(GAI_CUDA_WERROR)
        target_compile_options(gai_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>)
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# AMD HIP library
# ══════════════════════════════════════════════════════════════════════════════
if(GAI_HAVE_HIP)
    file(GLOB GAI_HIP_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/backends/backend_hip.cpp)
    add_library(gai_hip STATIC ${GAI_HIP_SOURCES})
    target_include_directories(gai_hip PUBLIC ${CMAKE_SOURCE_DIR})
    set_target_properties(gai_hip PROPERTIES HIP_STANDARD 17)
    target_compile_definitions(gai_hip PUBLIC GAI_HIP=1)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100")
        message(STATUS "[ghassan-ai] HIP archs: ${CMAKE_HIP_ARCHITECTURES} (override with -DCMAKE_HIP_ARCHITECTURES=...)")
    endif()
    if(hipblas_FOUND)
        target_link_libraries(gai_hip PUBLIC roc::hipblas)
    elseif(HIPBLAS_FOUND)
        target_include_directories(gai_hip PUBLIC ${HIPBLAS_INCLUDE_DIRS})
        target_link_libraries(gai_hip PUBLIC ${HIPBLAS_LIBRARIES})
    endif()
    target_link_libraries(gai_core PUBLIC gai_hip)
    target_compile_definitions(gai_core PUBLIC GAI_HIP=1)
    if(GAI_ENABLE_RCCL)
        find_library(RCCL_LIBRARY rccl HINTS /opt/rocm/lib)
        find_path(RCCL_INCLUDE rccl/rccl.h HINTS /opt/rocm/include)
        if(RCCL_LIBRARY AND RCCL_INCLUDE)
            target_include_directories(gai_core PUBLIC ${RCCL_INCLUDE})
            target_link_libraries(gai_core PUBLIC ${RCCL_LIBRARY})
            target_compile_definitions(gai_core PUBLIC GAI_RCCL=1)
            message(STATUS "[ghassan-ai] RCCL: multi-AMD-GPU enabled")
        else()
            message(STATUS "[ghassan-ai] RCCL: not found")
        endif()
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Intel SYCL library
# ══════════════════════════════════════════════════════════════════════════════
if(GAI_HAVE_SYCL)
    file(GLOB GAI_SYCL_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/backends/backend_sycl.cpp)
    add_library(gai_sycl STATIC ${GAI_SYCL_SOURCES})
    target_include_directories(gai_sycl PUBLIC ${CMAKE_SOURCE_DIR})
    target_compile_options(gai_sycl PRIVATE -fsycl)
    target_link_options(gai_sycl PRIVATE -fsycl)
    target_compile_definitions(gai_sycl PUBLIC GAI_SYCL=1)
    if(oneMKL_FOUND)
        target_link_libraries(gai_sycl PUBLIC MKL::MKL_SYCL)
    elseif(MKL_FOUND)
        target_link_libraries(gai_sycl PUBLIC ${MKL_LIBRARIES})
    endif()
    target_link_libraries(gai_core PUBLIC gai_sycl)
    target_compile_definitions(gai_core PUBLIC GAI_SYCL=1)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Google XLA library
# ══════════════════════════════════════════════════════════════════════════════
if(GAI_HAVE_XLA)
    file(GLOB GAI_XLA_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/backends/backend_xla*.cpp)
    add_library(gai_xla STATIC ${GAI_XLA_SOURCES})
    target_include_directories(gai_xla PUBLIC ${CMAKE_SOURCE_DIR} ${XLA_INCLUDE_DIR})
    target_compile_definitions(gai_xla PUBLIC GAI_XLA=1)
    target_link_libraries(gai_xla PUBLIC ${XLA_LIBRARY})
    if(TPU_LIBRARY)
        target_link_libraries(gai_xla PUBLIC ${TPU_LIBRARY})
        target_compile_definitions(gai_xla PUBLIC GAI_TPU=1)
    endif()
    target_link_libraries(gai_core PUBLIC gai_xla)
    target_compile_definitions(gai_core PUBLIC GAI_XLA=1)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# NCCL (NVIDIA multi-GPU)
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_NCCL OFF CACHE BOOL "NCCL multi-GPU support detected" FORCE)
if(GAI_HAVE_CUDA AND GAI_ENABLE_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include)
    find_library(NCCL_LIBRARY nccl
        HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        set(GAI_HAVE_NCCL ON CACHE BOOL "NCCL multi-GPU support detected" FORCE)
        target_include_directories(gai_core PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(gai_core PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(gai_core PUBLIC GAI_NCCL=1)
        message(STATUS "[ghassan-ai] NCCL: ${NCCL_LIBRARY} (multi-NVIDIA-GPU enabled)")
    else()
        message(STATUS "[ghassan-ai] NCCL: not found -> single-GPU only")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# MPI
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_MPI OFF CACHE BOOL "MPI multi-CPU support detected" FORCE)
if(GAI_ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_FOUND)
        set(GAI_HAVE_MPI ON CACHE BOOL "MPI multi-CPU support detected" FORCE)
        target_include_directories(gai_core PUBLIC ${MPI_INCLUDE_PATH})
        target_link_libraries(gai_core PUBLIC ${MPI_C_LIBRARIES})
        target_compile_definitions(gai_core PUBLIC GAI_MPI=1)
        message(STATUS "[ghassan-ai] MPI: ${MPI_C_LIBRARIES} (multi-CPU DDP enabled)")
    else()
        message(STATUS "[ghassan-ai] MPI: not found -> single-CPU only")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Arrow/Parquet
# ══════════════════════════════════════════════════════════════════════════════
set(GAI_HAVE_PARQUET OFF)
if(GAI_ENABLE_PARQUET)
    find_package(Arrow QUIET)
    find_package(Parquet QUIET)
    if(Arrow_FOUND AND Parquet_FOUND)
        set(GAI_HAVE_PARQUET ON)
        target_link_libraries(gai_core PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
        target_compile_definitions(gai_core PUBLIC GAI_PARQUET=1)
        message(STATUS "[ghassan-ai] Parquet: enabled")
    else()
        message(STATUS "[ghassan-ai] Parquet: GAI_ENABLE_PARQUET=ON but Arrow/Parquet not found")
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Platform-specific
# ══════════════════════════════════════════════════════════════════════════════
if(WIN32)
    target_compile_definitions(gai_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    if(NOT MSVC)
        target_link_libraries(gai_core PUBLIC ws2_32)
    endif()
endif()

if(NOT MSVC)
    find_package(Threads REQUIRED)
    target_link_libraries(gai_core PUBLIC Threads::Threads)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Executables
# ══════════════════════════════════════════════════════════════════════════════
add_executable(ghassan-ai      tools/inference_cli.cpp tools/model_loader.cpp tools/cli_common.cpp)
add_executable(gai_train       tools/training_cli.cpp  tools/cli_common.cpp tools/model_loader.cpp)
add_executable(gai_data        tools/data_prep_cli.cpp tools/cli_common.cpp tools/model_loader.cpp)
add_executable(gai_tiny_prepare tools/tiny_prepare.cpp)

foreach(t ghassan-ai gai_train gai_data gai_tiny_prepare)
    target_link_libraries(${t} PRIVATE gai_core)
    gai_apply_cxx_flags(${t})
endforeach()

# Build a small parity harness so tiled CUDA attention can be checked against
# the CPU online-softmax reference on an actual CUDA device (e.g. Kaggle T4).
if(GAI_HAVE_CUDA)
    add_executable(gai_cuda_attention_test tools/cuda_attention_test.cpp)
    target_link_libraries(gai_cuda_attention_test PRIVATE gai_core)
    gai_apply_cxx_flags(gai_cuda_attention_test)
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════╗")
message(STATUS "║           Ghassan AI — Build Summary                 ║")
message(STATUS "╠══════════════════════════════════════════════════════╣")
message(STATUS "║  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "║  CUDA (NVIDIA) : ${GAI_HAVE_CUDA}")
message(STATUS "║  HIP  (AMD)    : ${GAI_HAVE_HIP}")
message(STATUS "║  SYCL (Intel)  : ${GAI_HAVE_SYCL}")
message(STATUS "║  XLA  (TPU)    : ${GAI_HAVE_XLA}")
message(STATUS "║  NCCL multi-GPU: ${GAI_HAVE_NCCL}")
message(STATUS "║  native arch   : ${GAI_NATIVE_ARCH}  (AVX-512/VNNI/F16C)")
message(STATUS "╚══════════════════════════════════════════════════════╝")
message(STATUS "")
