cmake_minimum_required(VERSION 3.24)
project(qrh_sim_proj LANGUAGES CXX CUDA)

# --- Python + pybind11 from venv ---
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional; keep only if you truly need it)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# CUDA Toolkit (cudart + curand)
find_package(CUDAToolkit REQUIRED)

# --- CUDA arch: you can override this via -DCMAKE_CUDA_ARCHITECTURES=... ---
# If you want it fixed to your local GPU (RTX 2060 = 75), keep it.
# For AWS portability, you may later remove this and set it at configure time.
set(CMAKE_CUDA_ARCHITECTURES 75)

# ---------- Python extension module: qrh_sim_cuda ----------
pybind11_add_module(qrh_sim_cuda scripts/sim_paths_cuda.cu)

target_compile_features(qrh_sim_cuda PRIVATE cxx_std_17)

# CUDA-specific compile options (nvcc)
target_compile_options(qrh_sim_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG --expt-extended-lambda --expt-relaxed-constexpr>
)

# Host C++ compile options
if (MSVC)
  target_compile_options(qrh_sim_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /DNDEBUG>
  )
else()
  target_compile_options(qrh_sim_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
  )
endif()

set_target_properties(qrh_sim_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(qrh_sim_cuda PRIVATE CUDA::cudart CUDA::curand)

# Install module into site-packages root
install(TARGETS qrh_sim_cuda
  LIBRARY DESTINATION .
  ARCHIVE DESTINATION .
  RUNTIME DESTINATION .
)