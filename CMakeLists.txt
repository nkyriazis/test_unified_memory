project( test_unified_memory ) 
cmake_minimum_required( VERSION 3.0.2 )

find_package(CUDA)

list(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)
add_definitions(-D__cpp_lib_result_of_sfinae)
cuda_add_executable(test_unified_memory test.cu)