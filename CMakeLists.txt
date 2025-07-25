cmake_minimum_required(VERSION 3.10)

project(CauchyFriendly)

enable_language(CXX)

## Source Directory
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

## Runtime Library
set(CMAKE_INSTALL_RPATH /usr/lib64)

## Shared src Directory
set(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

## Set install location to bin
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH ${CMAKE_SOURCE_DIR} FORCE)

## include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)