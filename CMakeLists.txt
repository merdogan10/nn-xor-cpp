cmake_minimum_required(VERSION 3.9)
project(test VERSION 0.1
        DESCRIPTION "Neural Network"
        LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-O3")

include_directories(./libs/include)
include_directories(./layers)

set(TEST_TARGET testmain)

add_executable(${TEST_TARGET}
                main.cpp
                layers/neural_network.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

