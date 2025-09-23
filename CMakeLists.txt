cmake_minimum_required(VERSION 3.10)

project(CauchyFriendly)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 20) # We love spans
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library with implementation files
add_library(CauchyLibrary INTERFACE)
target_include_directories(CauchyLibrary
  INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(api)

# Testing
include(CTest)
add_subdirectory(test)
