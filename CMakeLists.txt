cmake_minimum_required(VERSION 3.20)
project(beam24 LANGUAGES CXX)

option(BEAM24_ENABLE_CUDA "Build Beam24 CUDA executables" ON)
option(BEAM24_ENABLE_CCGLIB "Build the pinned external ccglib system controls" OFF)
set(BEAM24_CUDA_ARCH "120a" CACHE STRING "CUDA architecture used for Beam24 kernels")

include(CheckLanguage)
if(BEAM24_ENABLE_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA requested but nvcc was not found. Set CUDACXX, or select the cpu-check preset explicitly.")
  endif()
  if(NOT BEAM24_CUDA_ARCH STREQUAL "120a")
    message(FATAL_ERROR "The maintained Beam24 CUDA path requires architecture 120a.")
  endif()
endif()

if(BEAM24_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  # CMake 4.x may otherwise add its detected default architecture (for example,
  # sm_75) alongside the explicit architecture required by these kernels.
  # The "a" architecture suffix is passed directly to nvcc because it denotes
  # architecture-specific features used by the SM120a implementation.
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  function(beam24_add_cuda_target target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_ARCHITECTURES OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3;-gencode=arch=compute_${BEAM24_CUDA_ARCH},code=sm_${BEAM24_CUDA_ARCH}>)
    target_link_libraries(${target} PRIVATE CUDA::cuda_driver CUDA::cudart)
  endfunction()

  beam24_add_cuda_target(beam24_operator src/cuda/beam24_operator.cu)
  beam24_add_cuda_target(beam24_system src/cuda/beam24_system.cu)
  beam24_add_cuda_target(
    beam24_hierarchical_system src/cuda/beam24_hierarchical_system.cu)
  beam24_add_cuda_target(
    beam24_hierarchical_quality src/cuda/beam24_hierarchical_quality.cu)
  target_compile_definitions(
    beam24_hierarchical_quality PRIVATE BEAM24_FINITE_QUALITY=1)
  target_link_libraries(beam24_hierarchical_quality PRIVATE CUDA::cublas)
  beam24_add_cuda_target(
    beam24_hierarchical_exhaustive src/cuda/beam24_hierarchical_system.cu)
  target_compile_definitions(
    beam24_hierarchical_exhaustive PRIVATE BEAM24_DEFAULT_MODE=0)
  beam24_add_cuda_target(beam24_dense_fused src/cuda/dense_fused_baseline.cu)
  beam24_add_cuda_target(
    beam24_cufft_streamed_top1
    baselines/cuda/cufft_streamed_uniform_angle_top1.cu)
  target_link_libraries(beam24_cufft_streamed_top1 PRIVATE CUDA::cufft)
  if(BEAM24_ENABLE_CCGLIB)
    if(NOT BEAM24_BASELINE_SOURCE_ROOT)
      message(FATAL_ERROR "Set BEAM24_BASELINE_SOURCE_ROOT using artifact/scripts/build_ccglib.sh")
    endif()
    foreach(dependency IN ITEMS ccglib cudawrappers xtl xtensor)
      if(NOT EXISTS "${BEAM24_BASELINE_SOURCE_ROOT}/${dependency}/CMakeLists.txt")
        message(FATAL_ERROR "Missing pinned baseline source: ${dependency}")
      endif()
    endforeach()
    set(FETCHCONTENT_SOURCE_DIR_CUDAWRAPPERS "${BEAM24_BASELINE_SOURCE_ROOT}/cudawrappers")
    set(FETCHCONTENT_SOURCE_DIR_XTL "${BEAM24_BASELINE_SOURCE_ROOT}/xtl")
    set(FETCHCONTENT_SOURCE_DIR_XTENSOR "${BEAM24_BASELINE_SOURCE_ROOT}/xtensor")
    set(CCGLIB_BACKEND CUDA CACHE STRING "" FORCE)
    set(CCGLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(CCGLIB_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(CCGLIB_BUILD_STATIC OFF CACHE BOOL "" FORCE)
    set(CUDAWRAPPERS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory("${BEAM24_BASELINE_SOURCE_ROOT}/ccglib" "ccglib-build" EXCLUDE_FROM_ALL)
    beam24_add_cuda_target(beam24_ccglib_hierarchy baselines/cuda/ccglib_hierarchical_top1.cu)
    beam24_add_cuda_target(beam24_ccglib_materialized_top1 baselines/cuda/ccglib_materialized_top1.cu)
    target_link_libraries(beam24_ccglib_hierarchy PRIVATE ccglib)
    target_link_libraries(beam24_ccglib_materialized_top1 PRIVATE ccglib)
  endif()
else()
  if(BEAM24_ENABLE_CCGLIB)
    message(FATAL_ERROR "External ccglib controls require BEAM24_ENABLE_CUDA=ON")
  endif()
  message(STATUS "Configured explicit CPU-only project checks")
endif()
