cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(libtorch_test)
find_package(Torch REQUIRED
            NO_MODULE
            PATHS E:\libtorch-win-shared-with-deps-2.7.0+cu128\libtorch
            NO_DEFAULT_PATH)
add_executable(MAIN main.cpp)
target_link_libraries(MAIN "${TORCH_LIBRARIES}")
set_property(TARGET MAIN PROPERTY CXX_STANDARD 14)