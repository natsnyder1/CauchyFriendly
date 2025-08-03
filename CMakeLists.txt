cmake_minimum_required(VERSION 3.10)

project(CauchyFriendly)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library with implementation files
add_library(CauchyLibrary INTERFACE)
target_include_directories(CauchyLibrary
  INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(main)

# ## Source Directory
# set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

# ## Runtime Library
# set(CMAKE_INSTALL_RPATH /usr/lib64)

# ## Shared src Directory
# set(SHARED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ## Set install location to bin
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH ${CMAKE_SOURCE_DIR} FORCE)

# ## include directories
# include_directories(
#   ${PROJECT_SOURCE_DIR}/include
# )

# # ## fmt 
# # cmake_path(APPEND PROJECT_SOURCE_DIR
# #     "external" "fmt"
# #     OUTPUT_VARIABLE fmt_BASE_DIR)
# # list(APPEND CMAKE_PREFIX_PATH ${fmt_BASE_DIR})
# # add_subdirectory(${fmt_BASE_DIR} EXCLUDE_FROM_ALL)
# # find_package(fmt)
# # include_directories(${fmt_BASE_DIR}/include)

# # # MuJoCo 2.3.1
# # set(mujoco_DIR ${PROJECT_SOURCE_DIR}/src/gtfo/External/mujoco/bin)
# # find_package(mujoco PATHS ${mujoco_DIR} REQUIRED)
# # include_directories(${mujoco_INCLUDE_DIR})

# # ## Generate compile commands for clangd
# # set(CMAKE_EXPORT_COMPILE_COMMANDS true)

# # ## For outputting something uncomment
# # # message(STATUS ${fmt_BASE_DIR})

# ## Build for Linux
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_C_FLAGS "-g3 -Wall -fPIC")
# set(CMAKE_CXX_FLAGS "-g3 -Wall -fPIC")
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#   set(CMAKE_CXX_FLAGS "-O3 -g3 -Wall -fPIC")
#   set(CMAKE_C_FLAGS "-O3 -g3 -Wall -fPIC")
# endif()

# # ## Point to Kawasaki library for KRNX
# # set(LIB_KRNX ${EXT_DIR}/kawasaki/krnx/lib/Linux-64bit/libkrnx-l64.so)

# # ## include GTFO
# # set(GTFO_BUILD_TESTS OFF)
# # add_subdirectory(${SHARED_SRC}/gtfo)

# # ## Sub-directories for executables - Uncomment lines to build them
# # add_subdirectory(${SHARED_SRC}/exe/krnx_api_sample)
# # add_subdirectory(${SHARED_SRC}/exe/krnx_RTC_sample)
# # add_subdirectory(${SHARED_SRC}/exe/single_arm_control)
# # add_subdirectory(${SHARED_SRC}/exe/force_sensor_test)
# # add_subdirectory(${SHARED_SRC}/exe/upper_limb_control)
# # add_subdirectory(${SHARED_SRC}/exe/gtfo_test)
# # add_subdirectory(${SHARED_SRC}/exe/upper_left_goto_testing_home)
# # add_subdirectory(${SHARED_SRC}/exe/upper_right_goto_testing_home)