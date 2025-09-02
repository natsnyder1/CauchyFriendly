# Main estimator API executable
add_executable(CauchyAPI cauchy_estimator_api.cpp)
target_link_libraries(CauchyAPI PUBLIC CauchyLibrary)
