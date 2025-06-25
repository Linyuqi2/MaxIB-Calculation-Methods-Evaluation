cmake_minimum_required(VERSION 3.14)
project(chebyshevball_evaluation)

option(DISABLE_NLP_ORACLES "Disable non-linear oracles (used in collocation)" ON)
if(DISABLE_NLP_ORACLES)
  add_definitions(-DDISABLE_NLP_ORACLES)
else()
  find_library(IFOPT NAMES libifopt_core.so PATHS /usr/local/lib)
  find_library(IFOPT_IPOPT NAMES libifopt_ipopt.so PATHS /usr/local/lib)
  find_library(GMP NAMES libgmp.so PATHS /usr/lib/x86_64-linux-gnu /usr/lib/i386-linux-gnu)
  find_library(MPSOLVE NAMES libmps.so PATHS /usr/local/lib)
  find_library(PTHREAD NAMES libpthread.so PATHS /usr/lib/x86_64-linux-gnu /usr/lib/i386-linux-gnu)
  find_library(FFTW3 NAMES libfftw3.so.3 PATHS /usr/lib/x86_64-linux-gnu /usr/lib/i386-linux-gnu)

  if (NOT IFOPT)

    message(FATAL_ERROR "This program requires the ifopt library, and will not be compiled.")

  elseif (NOT GMP)

    message(FATAL_ERROR "This program requires the gmp library, and will not be compiled.")

  elseif (NOT MPSOLVE)

    message(FATAL_ERROR "This program requires the mpsolve library, and will not be compiled.")

  elseif (NOT FFTW3)

    message(FATAL_ERROR "This program requires the fftw3 library, and will not be compiled.")

  else()
    message(STATUS "Library ifopt found: ${IFOPT}")
    message(STATUS "Library gmp found: ${GMP}")
    message(STATUS "Library mpsolve found: ${MPSOLVE}")
    message(STATUS "Library fftw3 found:" ${FFTW3})

  endif(NOT IFOPT)
endif(DISABLE_NLP_ORACLES)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# ————lp_solve————
include(${CMAKE_SOURCE_DIR}/external/volesti/external/cmake-files/LPSolve.cmake)
GetLPSolve()


# —— GLPK —— 
find_path(GLPK_INCLUDE_DIR
  NAMES glpk.h
  PATHS /usr/include /usr/local/include /usr/include/glpk
)

find_library(GLPK_LIBRARY
  NAMES glpk glpklib
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

if (NOT GLPK_INCLUDE_DIR OR NOT GLPK_LIBRARY)
  message(FATAL_ERROR 
    "Could not find GLPK (headers or library).  
     Please install the development package (e.g. libglpk-dev on Ubuntu) 
     or adjust CMakeLists.txt to point to your GLPK locations."
  )
endif()

include_directories(${GLPK_INCLUDE_DIR})

# ______ MOSEK_________
if(NOT DEFINED MOSEK_ROOT)
    if(DEFINED ENV{MOSEK_ROOT})
        set(MOSEK_ROOT $ENV{MOSEK_ROOT})
    else()
        message(FATAL_ERROR
            "请先设置环境变量 MOSEK_ROOT 或者在 cmake 时用 -DMOSEK_ROOT=<你的mosek路径> 指定")
    endif()
endif()
message(STATUS "Using MOSEK_ROOT = ${MOSEK_ROOT}")

set(MOSEK_INCLUDE_DIR "${MOSEK_ROOT}/tools/platform/linux64x86/h")
set(MOSEK_LIBRARY_DIR "${MOSEK_ROOT}/tools/platform/linux64x86/bin")

find_path(MOSEK_HDR mosek.h
    HINTS ${MOSEK_INCLUDE_DIR}
)
find_library(MOSEK_LIB mosek64
    HINTS ${MOSEK_LIBRARY_DIR}
)

if(NOT MOSEK_HDR OR NOT MOSEK_LIB)
    message(FATAL_ERROR 
        "在 ${MOSEK_INCLUDE_DIR} 找不到 mosek.h 或在 ${MOSEK_LIBRARY_DIR} 找不到 libmosek.so")
endif()

message(STATUS "Found MOSEK include: ${MOSEK_HDR}")
message(STATUS "Found MOSEK library: ${MOSEK_LIB}")

include_directories(
    ${MOSEK_HDR}
)

include_directories(
    /usr/include/eigen3
    external/volesti/include
    external/volesti/external
    include
)

add_executable(main
    src/main.cpp
    src/generate_polytope.cpp
    src/solve.cpp
    src/solve_mosek.cpp
    src/solve_other.cpp
    src/metrics.cpp
)

target_link_libraries(main 
    lp_solve
    ${GLPK_LIBRARY}
    ${MOSEK_LIB}
)
