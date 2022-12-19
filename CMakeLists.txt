cmake_minimum_required(VERSION 3.20)
project(cmake_and_cuda LANGUAGES CXX CUDA)
#set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_COMPILER g++-12)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.0/bin/nvcc")
set(CMAKE_CXX_FLAGS "lineinfo -O3 -Wall")
set(CMAKE_CUDA_FLAGS "-lineinfo -O3 -Wall --compiler-options \"-O3 -Wall\"")
#set_property(TARGET GLOBAL PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(TARGET GLOBAL PROPERTIES CUDA_ARCHITECTURES "75")
include(CTest)

#add_library(d_main STATIC
#  d_main.cu
#  d_main.h
#  CHECK.h
#  config.h
#  molecule.h
#  )
#
#add_library(h_main STATIC
#  h_main.cu
#  h_main.h
#  CHECK.h
#  config.h
#  molecule.h
#  )
#
#add_library(csvparser STATIC
#  csvparser.cu
#  csvparser.h
#  )
#
#add_library(csvwriter STATIC
#  csvwriter.cu
#  csvwriter.h
#  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
#target_compile_features(h_main PUBLIC cxx_std_11)
#target_compile_features(d_main PUBLIC cxx_std_11)
#target_compile_features(csvparser PUBLIC cxx_std_11)
#target_compile_features(csvwriter PUBLIC cxx_std_11)


# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
#set_target_properties( d_main
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#        )
#set_target_properties(d_main PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(h_main PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(csvparser PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(csvwriter PROPERTIES CUDA_ARCHITECTURES "75")
#


#add_executable(main main.cu main.h h_main.h d_main.h csvparser.h csvwriter.h molecule.h)
add_executable(main main.cu main.h h_main.h h_main.cu d_main.h d_main.cu csvparser.h csvparser.cu csvwriter.h csvwriter.cu molecule.h)

#target_link_libraries(main PRIVATE d_main h_main csvparser csvwriter)


set_property(TARGET main
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")

