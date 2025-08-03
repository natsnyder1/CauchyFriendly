set(TEST_SRCS
    test_coalign_with_zeros.cpp
    test_cpdf_ndim.cpp
    test_eig_solver.cpp
    test_enumerations.cpp
    test_rel_marg2d_cpdf.cpp
    test_target_tracking.cpp)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LAPACKE REQUIRED lapacke)

# Seems that my Ubuntu didn't ship a glpk.pc file
# pkg_check_modules(GLPK REQUIRED glpk)
find_path(GLPK_INCLUDE_DIR glpk.h
    HINTS /usr/include /usr/local/include)
find_library(GLPK_LIBRARY NAMES glpk
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

foreach(src IN LISTS TEST_SRCS)
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PUBLIC CauchyLibrary)
  target_include_directories(${name} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LAPACKE_INCLUDE_DIRS}
    ${GLPK_INCLUDE_DIR}
    )
  target_link_libraries(${name} PUBLIC
    ${LAPACKE_LIBRARIES}
    ${GLPK_LIBRARY}
    LAPACK::LAPACK 
    BLAS::BLAS
    )

  # Register with CTest so `ctest` can run them
  add_test(NAME ${name} COMMAND ${name})
endforeach()